import torch
from torch.utils.data import DataLoader
from model import TwoTowerModel
from torch.optim import AdamW
from ..dataset.interactions import AdBatch
from ..dataset.interactions import InteractionsDataset
from ..dataset.interactions import RawInteractionsDataset

class Trainer:
    def __init__(self,
                 train_epochs: int = 10,
                 batch_size: int = 64,
                 embedding_dim: int = 64, 
                 learning_rate: float = 0.001
    ):
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        train_dataset = InteractionsDataset(
            raw_interactions_dataset=RawInteractionsDataset()
        )

        train_dataloader = DataLoader(train_dataset, self.batch_size, shuffle=True)

        model = TwoTowerModel(ads_categorical_features=train_dataset.categorical_features, ads_hidden_dims=[1024, 256], n_users=train_dataset.n_users, embedding_dim=self.embedding_dim).to(self.device)

        optimizer = AdamW(model.parameters, lr=self.learning_rate)

        for epoch in range(self.train_epochs):
            for batch in train_dataloader:
                model_loss = model(batch)
                optimizer.zero_grad()
                model_loss.backward()
                optimizer.step()
        return model
    
if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
                




