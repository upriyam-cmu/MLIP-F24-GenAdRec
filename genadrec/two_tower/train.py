import os
import json
import numpy as np
import torch
from torch.nn import Module
from dataset.interactions import InteractionsDataset
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from two_tower.model import TwoTowerModel
from torch.optim import AdamW
from torch.optim import SparseAdam
from tqdm import tqdm
from typing import NamedTuple
from typing import Optional


class LoadedCheckpoint(NamedTuple):
    model: Module
    optimizer: Module
    sparse_optimizer: Module


class Trainer:
    def __init__(self,
                 train_epochs: int = 100,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 64,
                 embedding_dim: int = 64,
                 learning_rate: float = 0.001,
                 train_eval_every_n: int = 1,
                 save_model_every_n: int = 5,
                 max_grad_norm: int = 1,
                 force_dataset_reload: bool = False,
                 checkpoint_path: Optional[str] = None,
                 save_dir_root: str = "out/"
    ):
        self.train_epochs = train_epochs
        self.batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.train_eval_every_n = train_eval_every_n
        self.save_model_every_n = save_model_every_n
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

        self._init_dataset()

    def _init_dataset(self):
        self.train_dataset = InteractionsDataset(
            path="data/",
            is_train=True,
            force_reload=self.force_dataset_reload
        )
        
        self.eval_dataset = InteractionsDataset(
            path="data/",
            is_train=False,
            force_reload=self.force_dataset_reload
        )
    
    def from_pretrained(self,
                        path: str,
                        model: Module = None,
                        optimizer: Module = None,
                        sparse_optimizer: Module = None) -> LoadedCheckpoint:
        state = torch.load(path)
        if model is None:
            model = TwoTowerModel(ads_categorical_features=self.train_dataset.categorical_features, ads_hidden_dims=[1024, 512, 128], n_users=self.train_dataset.n_users, embedding_dim=self.embedding_dim, use_user_ids=True, device=self.device)
        
        model.load_state_dict(state["model"])
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])
        if sparse_optimizer is not None:
            sparse_optimizer.load_state_dict(state["sparse_optimizer"])
            
        return LoadedCheckpoint(model=model, optimizer=optimizer, sparse_optimizer=sparse_optimizer)

    def train(self):
        sampler = BatchSampler(RandomSampler(self.train_dataset), self.batch_size, False)
        train_dataloader = DataLoader(self.train_dataset, sampler=sampler, batch_size=None)

        self.model = TwoTowerModel(ads_categorical_features=self.train_dataset.categorical_features, ads_hidden_dims=[1024, 512, 128], n_users=self.train_dataset.n_users, embedding_dim=self.embedding_dim, use_user_ids=True, device=self.device)

        optimizer = AdamW(self.model.dense_grad_parameters(), lr=self.learning_rate)
        sparse_optimizer = SparseAdam(self.model.sparse_grad_parameters(), lr=self.learning_rate)

        if self.checkpoint_path is not None:
            checkpoint = self.from_pretrained(
                self.checkpoint_path,
                model=self.model,
                optimizer=optimizer,
                sparse_optimizer=sparse_optimizer
            )
            self.model, optimizer = checkpoint.model, checkpoint.optimizer
            sparse_optimizer = checkpoint.sparse_optimizer

        for epoch in range(self.train_epochs):
            self.model.train()
            training_losses = []
            with tqdm(train_dataloader, desc=f'Epoch {epoch+1}') as pbar:
                for batch in pbar:
                    model_loss = self.model(batch)
                    
                    optimizer.zero_grad()
                    sparse_optimizer.zero_grad()
                    
                    model_loss.backward()

                    optimizer.step()
                    sparse_optimizer.step()
                    
                    training_losses.append(model_loss.item())

                    pbar.set_postfix({'Loss': np.mean(training_losses[-50:])})
            
            if epoch % self.save_model_every_n == 0:
                state = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "sparse_optimizer": sparse_optimizer.state_dict()
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
            checkpoint = self.from_pretrained(self.checkpoint_path)
            model = checkpoint.model

        sampler = BatchSampler(RandomSampler(self.eval_dataset), self.eval_batch_size, False)
        eval_dataloader = DataLoader(self.eval_dataset, sampler=sampler, batch_size=None)
        eval_index = self.eval_dataset.get_index()

        metrics = None
        index_emb = self.model.ad_forward(eval_index)
        with tqdm(eval_dataloader, desc='Eval') as pbar:
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
    
    metrics = {} if metrics is None else metrics
    for k in ks:
        hits = (rank <= k).sum().item()
        if f"hr@{k}" in metrics:
            metrics[f"hr@{k}"] += hits
        else:
            metrics[f"hr@{k}"] = hits

    return metrics


if __name__ == "__main__":
    trainer = Trainer(
        learning_rate=0.0005,
        eval_batch_size=256,
        train_batch_size=32,
        embedding_dim=32,
        force_dataset_reload=False
    )
    trainer.train()
