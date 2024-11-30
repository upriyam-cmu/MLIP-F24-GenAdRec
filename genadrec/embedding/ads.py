import torch
from dataset.interactions import AdBatch
from dataset.interactions import CategoricalFeature
from model.mlp import build_mlp
from torch import nn
from typing import Iterable


class AdEmbedder(nn.Module):
    def __init__(self,
                 categorical_features: Iterable[CategoricalFeature],
                 embedding_dim: int,
                 device: torch.device
            ) -> None:
        super().__init__()

        self.embedding_modules = nn.ModuleDict({
            feat.name: nn.Embedding(feat.num_classes + (1 if feat.has_nulls else 0), embedding_dim, sparse=True, device=device) 
            for feat in categorical_features
        })
        self.feat_has_nulls = {feat.name: feat.has_nulls for feat in categorical_features}
        self.embedding_dim = embedding_dim
        self.device = device
    
    @property
    def out_dim(self):
        return self.embedding_dim*len(self.embedding_modules)

    def forward(self, batch: AdBatch):
        x = []
        for feat, id in batch._asdict().items():
            if feat in self.embedding_modules.keys():
                if self.feat_has_nulls[feat]:
                    id = id + 1
                x.append(self.embedding_modules[feat](id.to(torch.int32).to(self.device)))
        return torch.cat(x, axis=-1)


class AdTower(nn.Module):
    def __init__(self, categorical_features: Iterable[CategoricalFeature], embedding_dim, hidden_dims, device):
        super().__init__()
        self.device = device
        self.ad_embedder = AdEmbedder(categorical_features, embedding_dim, device=device)
        self.mlp = build_mlp(self.ad_embedder.out_dim, hidden_dims, embedding_dim).to(self.device)
    
    def forward(self, batch: AdBatch):
        emb = self.ad_embedder(batch)
        x = self.mlp(emb)
        return x
