import torch
import torch.nn as nn
from dataset.interactions import CategoricalFeature
from embedding.user import UserIdTower
from embedding.ads import AdTower
from itertools import chain
from loss.softmax import SampledSoftmaxLoss
from model.seq import RNN
from dataset.taobao_behavior_sequences import AdBatch, UserBatch
from dataset.taobao_behavior_sequences import TaobaoInteractionsSeqBatch
from typing import List


class RNNSeqModel(nn.Module):
    def __init__(self,
                 n_users: int,
                 n_actions: int,
                 user_categorical_feats: List[CategoricalFeature],
                 ad_categorical_feats: List[CategoricalFeature],
                 cell_type: str,
                 rnn_input_size,
                 rnn_hidden_size,
                 rnn_num_layers,
                 device,
                 embedder_hidden_dims,
                 rnn_batch_first=True,
                 use_random_negs=False,
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

        self.user_embedding = AdTower(
            categorical_features=user_categorical_feats,
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

        self.use_random_negs = use_random_negs
        self.action_embedding = nn.Embedding(n_actions+1, embedding_dim=rnn_input_size, max_norm=1, device=device)
        self.sampled_softmax = SampledSoftmaxLoss()
        self.device = device
    
    def dense_grad_parameters(self):
        return chain(
            self.action_embedding.parameters(),
            self.ad_embedding.mlp.parameters(),
            self.user_embedding.mlp.parameters(),
            self.rnn.parameters()
        )

    def sparse_grad_parameters(self):
        return chain(self.ad_embedding.ad_embedder.parameters(), self.user_embedding.ad_embedder.parameters())

    def forward(self, batch: TaobaoInteractionsSeqBatch):
        user_emb = self.user_embedding(batch.user_feats)
        ad_emb = self.ad_embedding(batch.ad_feats)
        action = batch.is_click + 2
        action_emb = self.action_embedding(action.to(self.device))
        is_click = batch.is_click == 1

        B, L, D = ad_emb.shape

        input_emb = ad_emb + action_emb

        shifted_is_click = is_click[:, 1:]
        
        self.rnn.reset()
        model_output = self.rnn(input_emb, user_emb.unsqueeze(0).repeat(2, 1, 1))[:, :-1, :]

        q_probas = batch.ad_feats.rel_ad_freqs.to(torch.float32).to(self.device)
        
        # Extract positive examples
        pos_ids = batch.ad_feats.adgroup_id[:, 1:][shifted_is_click].unsqueeze(1)
        target_emb = ad_emb[:, 1:, :][shifted_is_click, :]
        pos_emb = model_output[shifted_is_click, :]
        pos_q_probas = q_probas[:, 1:][shifted_is_click]
        
        # Extract negative examples
        if self.use_random_negs:
            neg_ids = batch.train_index.adgroup_id
            neg_emb = self.ad_forward(batch.train_index)
            neg_q_probas = batch.train_index.rel_ad_freqs.to(torch.float32).to(self.device)
            pos_neg_mask = (pos_ids != neg_ids).to(self.device)
        else:
            neg_ids = torch.flatten(batch.ad_feats.adgroup_id)
            neg_emb = torch.flatten(ad_emb, start_dim=0, end_dim=1)
            neg_q_probas = q_probas.flatten()
            pos_neg_mask = (
                (pos_ids != neg_ids) & ((~batch.is_padding).flatten())
            ).to(self.device)

        loss = self.sampled_softmax.forward(
            pos_emb=pos_emb,
            target_emb=target_emb,
            neg_emb=neg_emb,
            pos_q_probas=pos_q_probas,
            neg_q_probas=neg_q_probas,
            pos_neg_mask=pos_neg_mask.to(torch.float32)
        )

        return loss
    
    def eval_forward(self, batch: TaobaoInteractionsSeqBatch):
        user_emb = self.user_embedding(batch.user_feats)
        ad_emb = self.ad_embedding(batch.ad_feats)
        action = batch.is_click + 2
        action_emb = self.action_embedding(action.to(self.device))
        
        input_emb = ad_emb + action_emb
        self.rnn.reset()
        output_emb = self.rnn(input_emb, user_emb.unsqueeze(0).repeat(2,1,1))

        B, L, D = ad_emb.shape
        
        target_idx = (~batch.is_padding).sum(axis=1).unsqueeze(1) - 1
        batch_idx = torch.arange(B, device=ad_emb.device)
        target_emb = torch.diagonal(ad_emb[batch_idx, target_idx, :], dim1=0, dim2=1).T
        pred_emb = torch.diagonal(output_emb[batch_idx, target_idx-1, :], dim1=0, dim2=1).T

        return pred_emb, target_emb
    
    def ad_forward(self, batch: AdBatch):
        return self.ad_embedding(batch).squeeze(0)
    
    def user_forward(self, batch: UserBatch):
        return self.user_embedding(batch).squeeze(0)
