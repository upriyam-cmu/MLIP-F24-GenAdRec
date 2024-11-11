import torch.nn as nn
from dataset.interactions import CategoricalFeature
from embedding.user import UserIdTower
from embedding.ads import AdEmbedder
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

        self.ad_embedding = AdEmbedder(
            categorical_features=ad_categorical_feats,
            embedding_dim=rnn_input_size,
            device=device
        )

    def forward(self, batch: TaobaoInteractionsSeqBatch):
        user_emb = self.user_embedding(batch.user_feats)
        ad_emb = self.ad_embedding(batch.ad_feats)
        import pdb; pdb.set_trace()


