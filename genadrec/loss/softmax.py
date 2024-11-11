import torch
from torch import nn


class SampledSoftmaxLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pos_emb, target_emb, neg_emb, pos_q_probas, neg_q_probas, pos_neg_mask):
        pos_sim = torch.einsum("ij,ij->i", pos_emb, target_emb)
        neg_sim = torch.einsum("id,kd->ik", pos_emb, neg_emb)
        
        n_miss = pos_neg_mask.sum(axis=1).unsqueeze(1)
        neg_exp = torch.exp(
            neg_sim - torch.log(n_miss*neg_q_probas/(1-pos_q_probas.unsqueeze(1)) + 1e-10)
        )
        neg_exp = torch.einsum("ij,ij->i", pos_neg_mask, neg_exp)
        pos_exp = torch.exp(pos_sim)
        batch_loss = (-pos_sim + torch.log(pos_exp + neg_exp)).mean()
        return batch_loss
