import torch
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from two_tower.model import TwoTowerModel
from torch.optim import AdamW
from tqdm import tqdm
from dataset.interactions import InteractionsDataset
from dataset.interactions import RawInteractionsDataset


class Trainer:
    def __init__(self,
                 train_epochs: int = 10,
                 batch_size: int = 128,
                 embedding_dim: int = 64,
                 learning_rate: float = 0.001,
                 train_eval_every_n: int = 1
    ):
        self.train_epochs = train_epochs
        self.batch_size = batch_size
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

        self.model = TwoTowerModel(ads_categorical_features=self.train_dataset.categorical_features, ads_hidden_dims=[1024, 512], n_users=self.train_dataset.n_users, embedding_dim=self.embedding_dim).to(self.device)

        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

        for epoch in tqdm(range(self.train_epochs)):
            self.model.train()
            for batch in train_dataloader:
                model_loss = self.model(batch)
                optimizer.zero_grad()
                model_loss.backward()
                optimizer.step()
            
            if epoch % self.train_eval_every_n == 0:
                self.eval()
        return self.model
    
    @torch.inference_mode
    def eval(self):
        sampler = BatchSampler(RandomSampler(self.eval_dataset), self.batch_size, False)
        eval_dataloader = DataLoader(self.eval_dataset, sampler=sampler, batch_size=None)
        eval_index = self.eval_dataset.get_index()

        self.model.eval()
        for batch in eval_dataloader:
            user_emb, target_emb = self.model.user_forward(batch.user_feats), self.model.ad_forward(batch.ad_feats)
            index_emb = self.model.ad_forward(eval_index)
            
            metrics = compute_metrics(user_emb, target_emb, index_emb, ks=[1,5,10,50,100,200])
            print(metrics)
            

@torch.inference_mode
def compute_metrics(query, target, index, ks):
    B, D = query.shape
    q_t_sim = torch.einsum("ij,ij->i", query, target)
    q_i_sim = torch.einsum("ij,kj->ik", query, index)
    rank = (q_t_sim.unsqueeze(1) <= q_i_sim).sum(axis=1)
    
    metrics = {}
    for k in ks:
        metrics[f"hr@{k}"] = (rank <= k).sum().item() / B

    return metrics



if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()

