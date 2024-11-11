import torch
import torch.nn as nn
from dataset.interactions import CategoricalFeature
from embedding.user import UserIdTower
from embedding.ads import AdTower
from model.seq import RNN
from simple_ML_baseline.taobao_behavior_dataset import TaobaoInteractionsSeqBatch
from typing import List


class RNNSeqModel(nn.Module):
    def __init__(self,
                 n_users: int,
                 ad_categorical_feats: List[CategoricalFeature],
                 cell_type: str,
                 rnn_input_size,
                 rnn_hidden_size,
                 device,
                 embedder_hidden_dims,
                 rnn_batch_first=True,
                 ) -> None:
        
        super().__init__()
        
        self.rnn = RNN(
            cell_type=cell_type,
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            batch_first=rnn_batch_first
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

        self.action_embedding = nn.Embedding(3, embedding_dim=rnn_input_size, padding_idx=1, max_norm=1, device=device)

    def forward(self, batch: TaobaoInteractionsSeqBatch):
        user_emb = self.user_embedding(batch.user_feats)
        ad_emb = self.ad_embedding(batch.ad_feats)
        action = batch.is_click + 1
        action_emb = self.action_embedding(action)
        is_click = batch.is_click == 1

        input_emb = ad_emb + action_emb
        mask = (
            (~batch.is_padding).unsqueeze(1) &
            (batch.timestamp.unsqueeze(2) >= batch.timestamp.unsqueeze(1)) &
            (batch.ad_feats.adgroup_id.unsqueeze(2) != batch.ad_feats.adgroup_id.unsqueeze(1))
        )
    

        model_output = self.rnn(input_emb)[:, :-1, :]
        target_emb = ad_emb[:, 1:, :]

        import pdb; pdb.set_trace()

        B, L, D = target_emb.shape
        target_is_click = is_click[:, 1:, :]
        n_clicks = target_is_click.sum(axis=1).unsqueeze(1)
        click_mask = torch.arange(n_clicks.max(), device=n_clicks.device).repeat(B, 1) < n_clicks
        click_emb = torch.zeros(B, n_clicks.max(), D, device=n_clicks.device)
        click_emb[click_mask] = target_emb[target_is_click]

        is_click[:, 1:]


        import pdb; pdb.set_trace()
        

