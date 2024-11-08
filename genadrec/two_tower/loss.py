import torch
from torch import nn


class SampledSoftmaxLoss(nn.Module):
    def __init__(self, user_history_negs=True):
        super().__init__()
        self.user_history_negs = user_history_negs
    
    def forward(self, input_emb, target_emb, target_ids, user_ids, q_probas, mask, pos_mask):
        target_ids = target_ids.to(input_emb.device)
        mask = mask.to(input_emb.device)
        pos_mask = pos_mask.to(input_emb.device)
        q_probas = q_probas.to(input_emb.device)
        user_ids = user_ids.to(input_emb.device)

        pos_emb = input_emb[mask & pos_mask]
        pos_sim = torch.einsum("ij,ij->i", pos_emb, target_emb[mask & pos_mask])
        neg_sim = torch.einsum("id,kd->ik", pos_emb, target_emb[~mask])
        miss = (target_ids[mask & pos_mask].unsqueeze(1) != target_ids[~mask])

        if self.user_history_negs:
            same_user = (user_ids[mask & pos_mask].unsqueeze(1) == user_ids[~mask])
            same_user = (same_user | (same_user.sum(axis=1) == 0).unsqueeze(1))
            miss = (miss & same_user & ((~miss).sum(axis=0) == 0)).to(torch.float32)
            #miss *= ((1 - miss).sum(axis=0) == 0).to(torch.float32)
        
        n_miss = miss.sum(axis=1).unsqueeze(1)
        neg_exp = torch.exp(
            neg_sim - torch.log(n_miss*q_probas[~mask]/(1-q_probas[mask & pos_mask].unsqueeze(1)))
        )
        neg_exp = torch.einsum("ij,ij->i", miss, neg_exp)
        pos_exp = torch.exp(pos_sim)
        batch_loss = (-pos_sim + torch.log(pos_exp + neg_exp)).mean()
        #if batch_loss.item() < 8.4:
        import pdb; pdb.set_trace()
        return batch_loss
