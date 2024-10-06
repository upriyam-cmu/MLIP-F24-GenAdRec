import torch
from torch import nn


class SampledSoftmaxLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_emb, target_emb, target_ids, q_probas):
        sim = input_emb @ target_emb.T

        B = len(q_probas)
        miss = target_ids != target_ids.T
        
        n_miss = miss.sum(axis=1).unsqueeze(1)
        pos_sim = torch.diagonal(sim)
        neg_exp = (
            miss * (1 - torch.eye(B, device=sim.device)) *
            torch.exp(
                sim - torch.log(n_miss*q_probas/(1-q_probas.unsqueeze(1)))
            )
        )
        pos_exp = torch.exp(pos_sim)
        batch_loss = (-pos_sim + torch.log(pos_exp + neg_exp.sum(axis=1))).mean()
        return batch_loss