import torch
import torch.nn as nn
from dataset.interactions import CategoricalFeature
from embedding.user import UserIdTower
from embedding.ads import AdTower
from itertools import chain
from loss.softmax import SampledSoftmaxLoss
from model.seq import RNN
from simple_ML_baseline.taobao_behavior_dataset import AdBatch
from simple_ML_baseline.taobao_behavior_dataset import TaobaoInteractionsSeqBatch
from typing import List


class RNNSeqModel(nn.Module):
    def __init__(self,
                 n_users: int,
                 ad_categorical_feats: List[CategoricalFeature],
                 cell_type: str,
                 rnn_input_size,
                 rnn_hidden_size,
                 rnn_num_layers,
                 device,
                 embedder_hidden_dims,
                 rnn_batch_first=True,
                 ) -> None:
        
        super().__init__()
        
        self.rnn = RNN(
            cell_type=cell_type,
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            batch_first=rnn_batch_first,
            num_layers=rnn_num_layers,
            device=device
        )

        self.user_embedding = UserIdTower(
            n_users=n_users,
            embedding_dim=rnn_hidden_size,
            hidden_dims=embedder_hidden_dims,
            device=device
        )

        self.ad_embedding = AdTower(
            categorical_features=ad_categorical_feats,
            embedding_dim=rnn_input_size,
            hidden_dims=embedder_hidden_dims,
            device=device
        )

        self.rnn_num_layers = rnn_num_layers
        self.action_embedding = nn.Embedding(3, embedding_dim=rnn_input_size, padding_idx=1, max_norm=1, device=device)
        self.sampled_softmax = SampledSoftmaxLoss()
        self.device = device
    
    def dense_grad_parameters(self):
        return chain(
            self.action_embedding.parameters(),
            self.ad_embedding.mlp.parameters(),
            self.rnn.parameters()
        )

    def sparse_grad_parameters(self):
        return chain(self.ad_embedding.ad_embedder.parameters(), self.user_embedding.id_embedding.parameters())

    def forward(self, batch: TaobaoInteractionsSeqBatch):
        adgroup_id = batch.ad_feats.adgroup_id.to(self.device)
        user_emb = self.user_embedding(batch.user_feats)
        ad_emb = self.ad_embedding(batch.ad_feats)
        action = batch.is_click + 1
        action_emb = self.action_embedding(action.to(self.device))
        q_probas = batch.ad_feats.rel_ad_freqs.to(torch.float32).to(self.device)
        is_click = (batch.is_click == 1).to(self.device)
        is_padding = batch.is_padding.to(self.device)

        B, L, D = ad_emb.shape

        input_emb = ad_emb + action_emb

        shifted_is_click = is_click[:, 1:]
        seq_lengths = (~batch.is_padding).sum(axis=1)
        
        self.rnn.reset()
        model_output = self.rnn(input_emb, user_emb.unsqueeze(0).repeat(self.rnn_num_layers,1,1))[:, :-1, :]

        pos_ids = adgroup_id[:, 1:][shifted_is_click].unsqueeze(1)
        neg_ids = torch.flatten(adgroup_id)
        
        # user_expanded = batch.user_feats.user.unsqueeze(1).repeat(1,L)
        # pos_users = user_expanded[:,1:][shifted_is_click].unsqueeze(1)
        # neg_users = torch.flatten(user_expanded)
        #import pdb; pdb.set_trace()
        #START TEST BLOCK 
        
        has_clicks_after = shifted_is_click.flip(dims=[1]).cumsum(dim=1).flip(dims=[1]) != 0
        idxs = torch.arange(L-1, device=self.device).unsqueeze(0)
        click_pos = torch.where(shifted_is_click, idxs, torch.full_like(idxs, L-2))
        first_click_at_or_after = click_pos.flip(dims=[-1]).cummin(dim=-1).values.flip(dims=[-1])
        next_click_id = adgroup_id[:, 1:].gather(dim=1, index=first_click_at_or_after)
        next_click_emb = ad_emb[:, 1:].gather(dim=1, index=first_click_at_or_after.unsqueeze(2).repeat(1,1,D))
        pos_ids = next_click_id[has_clicks_after].unsqueeze(1)
        target_emb = next_click_emb[has_clicks_after]
        pos_emb = model_output[has_clicks_after]
        pos_q_probas = q_probas[:, 1:][has_clicks_after]

        # END TEST BLOCK

        
        #pos_neg_mask = torch.flatten(pos_neg_mask_expanded[shifted_is_click], start_dim=1, end_dim=2)
        pos_neg_mask = (
            (pos_ids != neg_ids) & ((~is_padding).flatten())
        )
        #target_emb = ad_emb[:, 1:, :][shifted_is_click, :]
        #pos_emb = model_output[shifted_is_click, :]
        neg_emb = torch.flatten(ad_emb, start_dim=0, end_dim=1)
        #pos_q_probas = q_probas[:, 1:][shifted_is_click]

        #import pdb; pdb.set_trace()

        loss = self.sampled_softmax.forward(
            pos_emb=pos_emb,
            target_emb=target_emb,
            neg_emb=neg_emb,
            pos_q_probas=pos_q_probas,
            neg_q_probas=q_probas.flatten(),
            pos_neg_mask=pos_neg_mask.to(torch.float32)
        )

        return loss
    
    def eval_forward(self, batch: TaobaoInteractionsSeqBatch):
        user_emb = self.user_embedding(batch.user_feats)
        ad_emb = self.ad_embedding(batch.ad_feats)
        action = batch.is_click + 1
        action_emb = self.action_embedding(action.to(self.device))
        
        input_emb = ad_emb + action_emb
        self.rnn.reset()
        output_emb = self.rnn(input_emb, user_emb.unsqueeze(0).repeat(self.rnn_num_layers,1,1))

        B, L, D = ad_emb.shape
        
        target_idx = (~batch.is_padding).sum(axis=1).unsqueeze(1) - 1
        batch_idx = torch.arange(B, device=ad_emb.device)
        target_emb = torch.diagonal(ad_emb[batch_idx, target_idx, :], dim1=0, dim2=1).T
        pred_emb = torch.diagonal(output_emb[batch_idx, target_idx-1, :], dim1=0, dim2=1).T

        return pred_emb, target_emb
    
    def ad_forward(self, batch: AdBatch):
        return self.ad_embedding(batch).squeeze(0)
        

