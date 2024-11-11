import torch
from dataset.interactions import AdBatch
from dataset.interactions import CategoricalFeature
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
