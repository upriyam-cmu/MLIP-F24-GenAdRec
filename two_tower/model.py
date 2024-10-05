import torch
from torch import nn
from torch import functional as F
from typing import NamedTuple
from typing import Iterable
from ..dataset.interactions import AdBatch
from ..dataset.interactions import CategoricalFeature
from ..dataset.interactions import InteractionsBatch
from ..dataset.interactions import UserBatch


class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


class UserTower(nn.Module):
    def __init__(self, n_users, embedding_dim):
        super().__init__()
        self.id_embedding = nn.Sequential(
            nn.Embedding(n_users, embedding_dim),
            L2NormalizationLayer(dim=-1)
        )
    
    def forward(self, batch: UserBatch):
        return self.id_embedding(batch.user)
    

class AdEmbedder(nn.Module):
    def __init__(self,
                 categorical_features: Iterable[CategoricalFeature],
                 embedding_dim: int
            ) -> None:
        super().__init__()

        self.embedding_modules = nn.ModuleDict({
            feat.name: nn.Embedding(feat.num_classes, embedding_dim) 
            for feat in categorical_features
        })
        
        self.embedding_dim = embedding_dim
    
    @property
    def out_dim(self):
        self.embedding_dim*len(self.embedding_modules)

    def forward(self, batch: AdBatch):
        x = []
        for feat, id in batch._asdict().items():
            x.append(self.embedding_modules[feat](id))
        return torch.cat(x, axis=-1)


class AdTower(nn.Module):
    def __init__(self, categorical_features: Iterable[CategoricalFeature], embedding_dim, hidden_dims):
        super().__init__()
        self.ad_embedder = AdEmbedder(categorical_features, embedding_dim)
        self.mlp = self.build_mlp(self.ad_embedder.out_dim, hidden_dims, embedding_dim)
    
    def build_mlp(self, in_dim, hidden_dims, out_dim):
        mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.SiLU()
        )

        for in_d, out_d in zip(hidden_dims[:-1], hidden_dims[1:]):
            mlp.add(nn.Linear(in_d, out_d))
            mlp.add(nn.SiLU())
        
        mlp.add(nn.Linear(hidden_dims[-1], out_dim))
        mlp.add(L2NormalizationLayer(dim=-1))
        return mlp

    def forward(self, batch):
        emb = self.ad_embedder(batch)
        x = self.mlp(emb)
        return x


class TwoTowerModel(nn.Module):
    def __init__(self, ads_categorical_features, ads_hidden_dims, n_users, embedding_dim):
        super().__init__(self)

        self.ad_tower = AdTower(categorical_features=ads_categorical_features, embedding_dim=embedding_dim, hidden_dims=ads_hidden_dims)
        self.user_tower = UserTower(n_users=n_users, embedding_dim=embedding_dim)
    
    def forward(self, batch):
        ad_embedding = self.ad_tower(batch.user_feats)
        user_embedding = self.user_tower(batch.ad_feats)
        import pdb; pdb.set_trace()

