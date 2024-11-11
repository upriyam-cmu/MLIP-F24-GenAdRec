import torch
from dataset.interactions import AdBatch
from dataset.interactions import InteractionsBatch
from embedding.ads import AdTower
from embedding.user import UserIdTower
from embedding.user import UserHistoryTower
from itertools import chain
from torch import nn
from two_tower.loss import SampledSoftmaxLoss


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
        return chain(self.ad_tower.mlp.parameters(), self.user_tower.mlp.parameters())
    
    def sparse_grad_parameters(self):
        if self.use_user_ids:
            return chain(self.user_tower.id_embedding.parameters(), self.ad_tower.ad_embedder.embedding_modules.parameters())
        return self.ad_tower.ad_embedder.embedding_modules.parameters()
    
    def forward(self, batch: InteractionsBatch):
        ad_embedding = self.ad_tower(batch.ad_feats).squeeze(0)
        if self.use_user_ids:
            user_embedding = self.user_tower(batch.user_feats).squeeze(0)
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
        #import pdb; pdb.set_trace()
        return batch_loss

    def eval_forward(self, batch: InteractionsBatch):
        ad_embedding = self.ad_tower(batch.ad_feats).squeeze(0)
        if self.use_user_ids:
            user_embedding = self.user_tower(batch.user_feats).squeeze(0)
        else:
            user_embedding = self.user_tower(batch, ad_embedding)
        return (user_embedding[batch.is_eval, :], ad_embedding[batch.is_eval, :])

    def ad_forward(self, batch: AdBatch):
        return self.ad_tower(batch).squeeze(0)
