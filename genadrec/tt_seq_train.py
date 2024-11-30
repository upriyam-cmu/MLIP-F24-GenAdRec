import os
import json
import numpy as np
import torch
from dataset.interactions import CategoricalFeature
from dataset.interactions import InteractionsDataset
from enum import Enum
from torch.nn import Module
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from two_tower.model import TwoTowerModel
from torch.optim import AdamW
from torch.optim import SparseAdam
from tqdm import tqdm
from typing import NamedTuple
from typing import Optional
from typing import List
from sequence_model.model import RNNSeqModel
from dataset.taobao_behavior_sequences import TaobaoSequenceDataset


class ModelType(Enum):
    TWO_TOWER = 1
    SEQ = 2


class Trainer:
    def __init__(self,
                 model_type: ModelType,
                 train_epochs: int = 100,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 64,
                 embedding_dim: int = 64,
                 learning_rate: float = 0.001,
                 train_eval_every_n: int = 1,
                 save_model_every_n: int = 5,
                 max_grad_norm: int = 1,
                 embedder_hidden_dims: Optional[List[int]] = [1024, 512, 128],
                 seq_rnn_cell_type: str = "GRU",
                 seq_rnn_num_layers: int = 2,
                 user_features: list[str] = ["gender", "age", "shopping", "occupation"],
                 ad_features: list[str] = ["cate", "brand"],
                 behavior_log_augmented: bool = False,
                 force_dataset_reload: bool = False,
                 checkpoint_path: Optional[str] = None,
                 save_dir_root: str = "out/"
    ):
        self.model_type = model_type
        self.train_epochs = train_epochs
        self.batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.train_eval_every_n = train_eval_every_n
        self.save_model_every_n = save_model_every_n
        self.embedder_hidden_dims = embedder_hidden_dims
        self.seq_rnn_cell_type = seq_rnn_cell_type
        self.seq_rnn_num_layers = seq_rnn_num_layers
        self.user_features = user_features
        self.ad_features = ad_features
        self.behavior_log_augmented = behavior_log_augmented
        self.force_dataset_reload = force_dataset_reload
        self.checkpoint_path = checkpoint_path
        self.save_dir_root = save_dir_root

        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        self.device = torch.device(device)

        self._init()

    def _init(self):
        if self.model_type == ModelType.TWO_TOWER:
            self.train_dataset = InteractionsDataset(
                path="raw_data/",
                is_train=True,
                force_reload=self.force_dataset_reload
            )

            sampler = BatchSampler(RandomSampler(self.train_dataset), self.batch_size, False)
            self.train_dataloader = DataLoader(self.train_dataset, sampler=sampler, batch_size=None)

            self.eval_dataset = InteractionsDataset(
                path="raw_data/",
                is_train=False,
                force_reload=self.force_dataset_reload
            )

            sampler = BatchSampler(RandomSampler(self.eval_dataset), self.eval_batch_size, False)
            self.eval_dataloader = DataLoader(self.eval_dataset, sampler=sampler, batch_size=None)

            self.model = TwoTowerModel(
                ads_categorical_features=self.train_dataset.categorical_features,
                ads_hidden_dims=self.embedder_hidden_dims,
                n_users=self.train_dataset.n_users,
                embedding_dim=self.embedding_dim,
                use_user_ids=True,
                device=self.device
            )

        elif self.model_type == ModelType.SEQ:
            self.train_dataset = TaobaoSequenceDataset(
                data_dir="data",
                is_train=True,
                augmented=self.behavior_log_augmented,
            )

            sampler = BatchSampler(RandomSampler(self.train_dataset), self.batch_size, False)
            self.train_dataloader = DataLoader(self.train_dataset, sampler=sampler, batch_size=None)

            self.eval_dataset = TaobaoSequenceDataset(
                data_dir="data",
                is_train=False,
                augmented=self.behavior_log_augmented,
            )

            sampler = BatchSampler(RandomSampler(self.eval_dataset), self.eval_batch_size, False)
            self.eval_dataloader = DataLoader(self.eval_dataset, sampler=sampler, batch_size=None)

            self.user_categorical_feats = [
                CategoricalFeature(
                    name=feat,
                    num_classes=self.train_dataset.user_encoder.feat_num_unique_with_null[feat]+1,
                    has_nulls=self.train_dataset.user_encoder.feat_has_null[feat]
                ) for feat in self.user_features
            ]
            self.ad_categorical_feats = [
                CategoricalFeature(
                    name=feat+"_id",
                    num_classes=self.train_dataset.ad_encoder.feat_num_unique_with_null[feat]+1,
                    has_nulls=self.train_dataset.ad_encoder.feat_has_null[feat],
                ) for feat in self.ad_features
            ]

            self.model = RNNSeqModel(
                n_users=self.train_dataset.n_users,
                n_actions=self.train_dataset.n_actions,
                user_categorical_feats=self.user_categorical_feats,
                ad_categorical_feats=self.ad_categorical_feats,
                cell_type=self.seq_rnn_cell_type,
                rnn_input_size=self.embedding_dim,
                rnn_hidden_size=self.embedding_dim,
                rnn_num_layers=self.seq_rnn_num_layers,
                device=self.device,
                embedder_hidden_dims=self.embedder_hidden_dims,
                use_random_negs=self.behavior_log_augmented,
                rnn_batch_first=True
            )

        else:
            raise Exception(f"Unsupported model type {self.model_type}")

        self.start_epoch = 0
        self.optimizer = AdamW(self.model.dense_grad_parameters(), lr=self.learning_rate)
        self.sparse_optimizer = SparseAdam(self.model.sparse_grad_parameters(), lr=self.learning_rate)


    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.start_epoch = state["epoch"] + 1
        self.optimizer.load_state_dict(state["optimizer"])
        self.sparse_optimizer.load_state_dict(state["sparse_optimizer"])
        self.model.load_state_dict(state["model"])


    def train(self):

        if self.checkpoint_path is not None:
            self.load_checkpoint(self.checkpoint_path)

        for epoch in range(self.start_epoch, self.train_epochs):
            self.model.train()
            training_losses = []
            with tqdm(self.train_dataloader, desc=f'Epoch {epoch+1}') as pbar:
                for batch in pbar:
                    model_loss = self.model(batch)

                    self.optimizer.zero_grad()
                    self.sparse_optimizer.zero_grad()

                    model_loss.backward()

                    self.optimizer.step()
                    self.sparse_optimizer.step()

                    training_losses.append(model_loss.item())
                    pbar.set_postfix({'Loss': np.mean(training_losses[-50:])})

            if epoch % self.save_model_every_n == 0:
                state = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "sparse_optimizer": self.sparse_optimizer.state_dict()
                }

                if not os.path.exists(self.save_dir_root):
                    os.makedirs(self.save_dir_root)

                torch.save(state, self.save_dir_root + f"checkpoint_{epoch}.pt")

            if epoch % self.train_eval_every_n == 0:
                self.eval(self.model, save_dir=self.save_dir_root + f"eval_{epoch}/")

        return self.model

    @torch.inference_mode
    def eval(self, model: Module = None, save_dir: str = None):
        assert model is not None or self.checkpoint_path is not None, "Model and checkpoint are both None"

        if save_dir is None:
            save_dir = self.save_dir_root + "eval/"

        if model is None:
            checkpoint = self.load_checkpoint(self.checkpoint_path)
            model = checkpoint.model

        eval_index = self.eval_dataset.get_index()

        metrics = None
        index_emb = model.ad_forward(eval_index)
        with tqdm(self.eval_dataloader, desc='Eval') as pbar:
            for batch in pbar:
                user_emb, target_emb = model.eval_forward(batch)

                metrics = accumulate_metrics(user_emb, target_emb, index_emb, ks=[1,10,50,100,200,500], metrics=metrics)

        metrics = {k: (v/len(self.eval_dataset)) for k, v in metrics.items()}

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(save_dir + "eval_metrics.json", "w") as f:
            json.dump(metrics, f)

        print(metrics)



@torch.inference_mode
def accumulate_metrics(query, target, index, ks, metrics=None):
    q_t_sim = torch.einsum("ij,ij->i", query, target)
    q_i_sim = torch.einsum("ij,kj->ik", query, index)
    rank = (q_t_sim.unsqueeze(1) <= q_i_sim).sum(axis=1)

    # compute and return ndcg
    ndcg_score = np.log(2) / torch.log(rank + 1)  # for single target, ndcg expression can be simplified
    ndcg_score = torch.clip(ndcg_score, 0.0, 1.0)  # clip just in case

    metrics = {} if metrics is None else metrics
    for k in ks:
        hits = (rank <= k).sum().item()
        if f"hr@{k}" in metrics:
            metrics[f"hr@{k}"] += hits
        else:
            metrics[f"hr@{k}"] = hits

    if "ndcg" in metrics:
        metrics["ndcg"] += ndcg_score.sum().item()
    else:
        metrics["ndcg"] = ndcg_score.sum().item()

    return metrics


if __name__ == "__main__":
    trainer = Trainer(
        model_type=ModelType.SEQ,
        learning_rate=0.001, # 0.0005 for two_tower
        eval_batch_size=1024,
        train_batch_size=32,
        embedding_dim=128,
        embedder_hidden_dims=[128],
        force_dataset_reload=False,
        save_model_every_n=1,
        train_eval_every_n=1,
        user_features=["gender", "age", "shopping", "occupation"],
        ad_features=["cate", "brand"],
        behavior_log_augmented=False,
        # save_dir_root="out_128_aug_uid_only/"
        # checkpoint_path="out/checkpoint_0.pt"
    )
    print("Model size:", sum(param.numel() for param in trainer.model.parameters()))
    # trainer.eval()
    trainer.train()
