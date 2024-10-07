import numpy as np
import torch
from dataset.interactions import InteractionsDataset
from dataset.interactions import RawInteractionsDataset
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from two_tower.model import TwoTowerModel
from torch.optim import AdamW
from torch.optim import SparseAdam
from tqdm import tqdm
import time


class Trainer:
    def __init__(self,
                 train_epochs: int = 10,
                 train_batch_size: int = 2048,
                 eval_batch_size: int = 64,
                 embedding_dim: int = 64,
                 learning_rate: float = 0.001,
                 train_eval_every_n: int = 1
    ):
        self.train_epochs = train_epochs
        self.batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.train_eval_every_n = train_eval_every_n
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._init_dataset()

    def _init_dataset(self):
        raw_dataset = RawInteractionsDataset()
        
        self.train_dataset = InteractionsDataset(
            raw_interactions_dataset=raw_dataset,
            is_train=True
        )
        
        self.eval_dataset = InteractionsDataset(
            raw_interactions_dataset=raw_dataset,
            is_train=False
        )

    def train(self):
        sampler = BatchSampler(RandomSampler(self.train_dataset), self.batch_size, False)
        train_dataloader = DataLoader(self.train_dataset, sampler=sampler, batch_size=None)

        self.model = TwoTowerModel(ads_categorical_features=self.train_dataset.categorical_features, ads_hidden_dims=[1024, 512], n_users=self.train_dataset.n_users, embedding_dim=self.embedding_dim, device=self.device)

        optimizer = AdamW(self.model.dense_grad_parameters(), lr=self.learning_rate)
        sparse_optimizer = SparseAdam(self.model.sparse_grad_parameters(), lr=self.learning_rate)

        self.eval()

        for epoch in range(self.train_epochs):
            self.model.train()
            training_losses = []
            with tqdm(train_dataloader, desc=f'Epoch {epoch+1}') as pbar:
                for batch in pbar:
                    model_loss = self.model(batch)
                    start = time.time()
                    optimizer.zero_grad()
                    model_loss.backward()
                    optimizer.step()
                    sparse_optimizer.step()
                    end = time.time()
                    #print(f"Backward time: {end - start}")
                    
                    training_losses.append(model_loss.item())
                    pbar.set_postfix({'Loss': np.mean(training_losses[-100:])})
            
            if epoch % self.train_eval_every_n == 0:
                self.eval()
        return self.model
    
    @torch.inference_mode
    def eval(self):
        sampler = BatchSampler(RandomSampler(self.eval_dataset), self.eval_batch_size, False)
        eval_dataloader = DataLoader(self.eval_dataset, sampler=sampler, batch_size=None)
        eval_index = self.eval_dataset.get_index()

        self.model.eval()
        metrics = None
        with tqdm(eval_dataloader, desc=f'Eval') as pbar:
            for batch in pbar:
                user_emb, target_emb = self.model.user_forward(batch.user_feats), self.model.ad_forward(batch.ad_feats)
                index_emb = self.model.ad_forward(eval_index)
                
                metrics = accumulate_metrics(user_emb, target_emb, index_emb, ks=[1,5,10,50,100,200], metrics=metrics)
        
        metrics = {k: (v/len(self.eval_dataset)) for k, v in metrics.items()}
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
    trainer = Trainer()
    trainer.train()

