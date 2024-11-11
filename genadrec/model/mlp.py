import torch.nn as nn


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