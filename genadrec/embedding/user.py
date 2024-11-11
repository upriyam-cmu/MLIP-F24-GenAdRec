import torch
from dataset.interactions import InteractionsBatch
from dataset.interactions import UserBatch
from model.mlp import L2NormalizationLayer
from model.mlp import build_mlp
from torch import nn


class UserIdTower(nn.Module):
    def __init__(self, n_users, embedding_dim, hidden_dims, device):
        super().__init__()
        self.id_embedding = nn.Sequential(
            nn.Embedding(n_users, embedding_dim, sparse=True, device=device),
            L2NormalizationLayer(dim=-1)
        )
        self.mlp = build_mlp(embedding_dim, hidden_dims, embedding_dim).to(device)
        self.device = device
    
    def forward(self, batch: UserBatch):
        emb = self.id_embedding(batch.user.to(self.device))
        x = self.mlp(emb)
        return x


class UserHistoryTower(nn.Module):
    def __init__(self, embedding_dim, hidden_dims, device):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = build_mlp(embedding_dim, hidden_dims, embedding_dim).to(device)
        self.device = device

    def forward(self, batch: InteractionsBatch, ad_embeddings: torch.Tensor):
        user_matches = (batch.user_feats.user.T == batch.user_feats.user)
        causal_mask = (batch.timestamp.unsqueeze(1) > batch.timestamp.unsqueeze(0))
        mask = (batch.is_click * user_matches * causal_mask).to(ad_embeddings.dtype).to(ad_embeddings.device)
        norm_mask = mask / (mask.sum(axis=1).unsqueeze(1) + 1e-16)
        #history_emb = torch.einsum("ij,jk->ik", norm_mask, ad_embeddings)
        history_emb = torch.einsum("ij,jk->ik", norm_mask, ad_embeddings)
        #x[torch.norm(history_emb, dim=1) == 0, :] = 0
        #import pdb; pdb.set_trace()
        return history_emb