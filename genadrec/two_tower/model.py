import torch
from dataset.interactions import AdBatch
from dataset.interactions import CategoricalFeature
from dataset.interactions import UserBatch
from torch import nn
from two_tower.loss import SampledSoftmaxLoss
from typing import Iterable


class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return nn.functional.normalize(x, p=2, dim=self.dim, eps=self.eps)


class UserTower(nn.Module):
    def __init__(self, n_users, embedding_dim, device):
        super().__init__()
        self.id_embedding = nn.Sequential(
            nn.Embedding(n_users, embedding_dim, device=device),
            L2NormalizationLayer(dim=-1)
        )
        self.device = device
    
    def forward(self, batch: UserBatch):
        return self.id_embedding(batch.user.to(self.device))
    

class AdEmbedder(nn.Module):
    def __init__(self,
                 categorical_features: Iterable[CategoricalFeature],
                 embedding_dim: int,
                 device: torch.device
            ) -> None:
        super().__init__()

        self.embedding_modules = nn.ModuleDict({
            feat.name: nn.Embedding(feat.num_classes, embedding_dim, device=device) 
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
        self.mlp = self.build_mlp(self.ad_embedder.out_dim, hidden_dims, embedding_dim)
    
    def build_mlp(self, in_dim, hidden_dims, out_dim):
        mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.SiLU()
        )

        for in_d, out_d in zip(hidden_dims[:-1], hidden_dims[1:]):
            mlp.append(nn.Linear(in_d, out_d))
            mlp.append(nn.SiLU())
        
        mlp.append(nn.Linear(hidden_dims[-1], out_dim))
        mlp.append(L2NormalizationLayer(dim=-1))
        return mlp.to(self.device)

    def forward(self, batch):
        emb = self.ad_embedder(batch)
        x = self.mlp(emb)
        return x


class TwoTowerModel(nn.Module):
    def __init__(self, ads_categorical_features, ads_hidden_dims, n_users, embedding_dim, device):
        super().__init__()

        self.ad_tower = AdTower(categorical_features=ads_categorical_features, embedding_dim=embedding_dim, hidden_dims=ads_hidden_dims, device=device)
        self.user_tower = UserTower(n_users=n_users, embedding_dim=embedding_dim, device=device)
        self.sampled_softmax = SampledSoftmaxLoss()
        self.device = device
    
    def forward(self, batch):
        ad_embedding = self.ad_tower(batch.ad_feats).squeeze(0)
        user_embedding = self.user_tower(batch.user_feats).squeeze(0)

        # In-batch softmax. Maybe TODO: Use random index
        batch_loss = self.sampled_softmax.forward(
            user_embedding,
            ad_embedding,
            batch.ad_feats.adgroup_id,
            batch.ad_feats.q_proba.flatten()
        )

        return batch_loss

    def user_forward(self, batch):
        user_embedding = self.user_tower(batch).squeeze(0)
        return user_embedding
    
    def ad_forward(self, batch):
        ad_embedding = self.ad_tower(batch).squeeze(0)
        return ad_embedding