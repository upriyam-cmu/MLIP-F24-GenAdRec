import torch
from torch import nn


class SampledSoftmaxLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_emb, target_emb, target_ids, q_probas, mask):
        target_ids = target_ids.to(input_emb.device)
        mask = mask.to(input_emb.device)
        q_probas = q_probas.to(input_emb.device)

        pos_emb = input_emb[mask]
        pos_sim = torch.einsum("ij,ij->i", pos_emb, target_emb[mask])
        neg_sim = pos_emb @ target_emb[~mask].T
        miss = (target_ids[mask].unsqueeze(1) != target_ids[~mask]).to(torch.float32)
        
        n_miss = miss.sum(axis=1).unsqueeze(1)
        neg_exp = torch.exp(
            neg_sim - torch.log(n_miss*q_probas[~mask]/(1-q_probas[mask].unsqueeze(1)))
        )
        neg_exp = torch.einsum("ij,ij->i", miss, neg_exp)
        pos_exp = torch.exp(pos_sim)
        batch_loss = (-pos_sim + torch.log(pos_exp + neg_exp)).mean()
        return batch_loss
