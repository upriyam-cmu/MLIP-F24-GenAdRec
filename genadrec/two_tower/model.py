import torch
from dataset.interactions import AdBatch
from dataset.interactions import CategoricalFeature
from dataset.interactions import InteractionsBatch
from itertools import chain
from torch import nn
from two_tower.loss import SampledSoftmaxLoss
from typing import Iterable
import time

def build_mlp(in_dim, hidden_dims, out_dim):
    mlp = nn.Sequential(
        nn.Linear(in_dim, hidden_dims[0]),
        nn.SiLU()
    )

    for in_d, out_d in zip(hidden_dims[:-1], hidden_dims[1:]):
        mlp.append(nn.Linear(in_d, out_d))
        mlp.append(nn.SiLU())
    
    mlp.append(nn.Linear(hidden_dims[-1], out_dim))
    mlp.append(L2NormalizationLayer(dim=-1))
    return mlp


class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-16):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return nn.functional.normalize(x, p=2, dim=self.dim, eps=self.eps)


class UserIdTower(nn.Module):
    def __init__(self, n_users, embedding_dim, hidden_dims, device):
        super().__init__()
        self.id_embedding = nn.Sequential(
            nn.Embedding(n_users, embedding_dim, sparse=True, device=device),
            L2NormalizationLayer(dim=-1)
        )
        self.mlp = build_mlp(embedding_dim, hidden_dims, embedding_dim).to(device)
        self.device = device
    
    def forward(self, batch: InteractionsBatch):
        emb = self.id_embedding(batch.user_feats.user.to(self.device))
        x = self.mlp(emb)
        return x


class UserHistoryTower(nn.Module):
    def __init__(self, embedding_dim, hidden_dims, device):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = build_mlp(embedding_dim, hidden_dims, embedding_dim).to(device)
        self.norm = L2NormalizationLayer(dim=-1)
        self.device = device

    def forward(self, batch: InteractionsBatch, ad_embeddings: torch.Tensor):
        user_matches = (batch.user_feats.user.T == batch.user_feats.user)
        causal_mask = (batch.timestamp.unsqueeze(1) > batch.timestamp.unsqueeze(0))
        mask = (batch.is_click * user_matches * causal_mask).to(ad_embeddings.dtype)
        norm_mask = mask / (mask.sum(axis=1).unsqueeze(1) + 1e-16)
        history_emb = torch.einsum("ij,jk->ik", norm_mask, ad_embeddings)
        x = self.norm(self.mlp(history_emb))
        x[torch.norm(history_emb, dim=1) == 0, :] = 0
        return x
    

class AdEmbedder(nn.Module):
    def __init__(self,
                 categorical_features: Iterable[CategoricalFeature],
                 embedding_dim: int,
                 device: torch.device
            ) -> None:
        super().__init__()

        self.embedding_modules = nn.ModuleDict({
            feat.name: nn.Embedding(feat.num_classes, embedding_dim, sparse=True, device=device) 
            for feat in categorical_features
        })
        
        self.embedding_dim = embedding_dim
        self.device = device
    
    @property
    def out_dim(self):
        return self.embedding_dim*len(self.embedding_modules)

    def forward(self, batch: AdBatch):
        x = []
        for feat, id in batch._asdict().items():
            if feat in self.embedding_modules.keys():
                x.append(self.embedding_modules[feat](id.to(torch.int32).to(self.device)))
        return torch.cat(x, axis=-1)


class AdTower(nn.Module):
    def __init__(self, categorical_features: Iterable[CategoricalFeature], embedding_dim, hidden_dims, device):
        super().__init__()
        self.device = device
        self.ad_embedder = AdEmbedder(categorical_features, embedding_dim, device=device)
        self.mlp = build_mlp(self.ad_embedder.out_dim, hidden_dims, embedding_dim).to(self.device)
        self.norm = L2NormalizationLayer(dim=-1)
    
    def forward(self, batch: AdBatch):
        emb = self.ad_embedder(batch)
        x = self.norm(self.mlp(emb))
        return x


class TwoTowerModel(nn.Module):
    def __init__(self, ads_categorical_features, ads_hidden_dims, n_users, embedding_dim, device, use_user_ids=False):
        super().__init__()

        self.ad_tower = AdTower(categorical_features=ads_categorical_features, embedding_dim=embedding_dim, hidden_dims=ads_hidden_dims, device=device)
        self.user_tower = (
            UserIdTower(n_users=n_users, embedding_dim=embedding_dim, hidden_dims=ads_hidden_dims, device=device) if use_user_ids
            else UserHistoryTower(embedding_dim=embedding_dim, hidden_dims=ads_hidden_dims, device=device)
        )
        self.use_user_ids = use_user_ids
        self.sampled_softmax = SampledSoftmaxLoss()
        self.device = device

    def dense_grad_parameters(self):
        if self.use_user_ids:
            return chain(self.ad_tower.mlp.parameters(), self.user_tower.mlp.parameters())
        return self.ad_tower.mlp.parameters()
    
    def sparse_grad_parameters(self):
        if self.use_user_ids:
            return chain(self.user_tower.id_embedding.parameters(), self.ad_tower.ad_embedder.embedding_modules.parameters())
        return self.ad_tower.ad_embedder.embedding_modules.parameters()
    
    def forward(self, batch: InteractionsBatch):
        ad_embedding = self.ad_tower(batch.ad_feats).squeeze(0)
        if self.use_user_ids:
            user_embedding = self.user_tower(batch).squeeze(0)
        else:
            user_embedding = self.user_tower(batch, ad_embedding)
        # In-batch softmax. Maybe TODO: Use random index
        batch_loss = self.sampled_softmax.forward(
            user_embedding,
            ad_embedding,
            batch.ad_feats.adgroup_id.squeeze(0).to(torch.int32),
            batch.user_feats.user.squeeze(0).to(torch.int32),
            batch.ad_feats.q_proba.squeeze(0).to(torch.float32),
            batch.is_click == 1,
            torch.norm(user_embedding, dim=1) != 0
        )
        #print(f"Forward: {end - start}")
        return batch_loss

    def eval_forward(self, batch: InteractionsBatch):
        ad_embedding = self.ad_tower(batch).squeeze(0)[batch.is_eval, :]
        user_embedding = self.user_tower(batch, ad_embedding)[batch.is_eval, :]
        return (user_embedding, ad_embedding)

    def ad_forward(self, batch: AdBatch):
        return self.ad_tower(batch).squeeze(0)
