import numpy as np
import os
import torch
import torch.nn.functional as F

from dataset.privacy_dataset import UserFeaturesDataset
from model.mlp import build_mlp
from torch import nn
from torch.optim import AdamW
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
)
from tt_seq_train import Trainer
from tqdm import tqdm

class PrivacyTrainer:
    def __init__(
        self,
        get_user_embeddings_fn,
        mlp_input_dim: int,
        save_dir: str,
        model_checkpoint = None,
        learning_rate = 0.001,
        train_epochs: int = 100,
        train_batch_size: int = 32,
        save_model_every_n: int = 1,
        mlp_hidden_dims: list[int] = [128, 64, 32],
        predict_user_features: list[str] = ["gender", "age", "occupation"],
    ):
        self.save_dir = save_dir
        self.learning_rate = learning_rate
        self.train_epochs = train_epochs
        self.train_batch_size = train_batch_size
        self.save_model_every_n = save_model_every_n
        self.model_checkpoint = model_checkpoint
        self.predict_user_features = predict_user_features
        
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        self.device = torch.device(device)
        
        self.dataset = UserFeaturesDataset("data")
        sampler = BatchSampler(RandomSampler(self.dataset), self.train_batch_size, False)
        self.dataloader = DataLoader(self.dataset, sampler=sampler, batch_size=None)

        self.get_user_embeddings = get_user_embeddings_fn
        self.model = nn.ModuleDict({
            feat: build_mlp(
                in_dim=mlp_hidden_dims[0], 
                hidden_dims=mlp_hidden_dims[1:],
                out_dim=self.dataset.user_encoder.feat_num_unique_with_null[feat],
                normalize=False,
            ).to(self.device)
            for feat in self.predict_user_features
        })
        self.model["base"] = nn.Linear(mlp_input_dim, mlp_hidden_dims[0], device=self.device)

        self.start_epoch = 0
        self.training_losses = []
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        if self.model_checkpoint is not None:
            self.load_checkpoint()


    def load_checkpoint(self, path: str = None) -> None:
        if path is None:
            assert self.model_checkpoint is not None
            path = os.path.join(self.save_dir, self.model_checkpoint)
        state = torch.load(path, map_location=self.device)
        self.start_epoch = state["epoch"] + 1
        self.training_losses = state["training_losses"]
        self.optimizer.load_state_dict(state["optimizer"])
        self.model.load_state_dict(state["model"])

        
    def train(self):
        for epoch in range(self.start_epoch, self.train_epochs):
            self.model.train()
            with tqdm(self.dataloader, desc=f'Epoch {epoch+1}') as pbar:
                for batch in pbar:
                    user_embeddings = self.get_user_embeddings(batch)
                    base_layer_out = self.model["base"](user_embeddings)
                    loss = sum(
                        F.cross_entropy(
                            input=self.model[feat](base_layer_out),
                            target=getattr(batch, feat).to(self.device, torch.int64),
                        )
                        for feat in self.predict_user_features
                    )
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.training_losses.append(loss.item())
                    pbar.set_postfix({'Loss': np.mean(self.training_losses[-50:])})

            if epoch % self.save_model_every_n == 0:
                state = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "losses": self.training_losses,
                    "accuracy": self.check_accuracy()
                }

                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

                torch.save(state, os.path.join(self.save_dir, f"checkpoint_{epoch}.pt"))


    @torch.inference_mode
    def check_accuracy(self):
        correct = 0
        total = 0
        for batch in (pbar := tqdm(self.dataloader, desc=f'Testing Accuracy')):
            user_embeddings = self.get_user_embeddings(batch)
            base_layer_out = self.model["base"](user_embeddings)

            B = user_embeddings.shape[0]
            total += B

            batch_correct = torch.ones(B, dtype=torch.bool, device=self.device)
            for feat in self.predict_user_features:
                logits = self.model[feat](base_layer_out)
                target = getattr(batch, feat).to(self.device).long()
                batch_correct = torch.logical_and(logits.argmax(dim=1) == target, batch_correct)
            
            correct += batch_correct.sum().item()
            pbar.set_postfix({'Accuracy': correct / total})

        return correct / total


if __name__ == "__main__":
    uids_only = True
    checkpoint_folder = "out_128_aug_uid_only"
    checkpoint_no = 5
    
    checkpoint = os.path.join(checkpoint_folder, f"checkpoint_{checkpoint_no}.pt")
    print(">>> Loading user embedding model:", checkpoint)
    trainer = Trainer(
        user_features=["user"] if uids_only else ["gender", "age", "shopping", "occupation"],
        checkpoint_path = checkpoint,
        behavior_log_augmented = True,
    )
    print(">>> Finished loading user embedding model:", checkpoint)

    print(">>> Creating user feature extraction head and loading user dataset")
    privacy_model_trainer = PrivacyTrainer(
        get_user_embeddings_fn=trainer.model.user_forward,
        mlp_input_dim=trainer.embedding_dim,
        mlp_hidden_dims=[64, 32],
        save_dir = os.path.join(checkpoint_folder, f"eval_{checkpoint_no}", "privacy")
    )
    print(">>> Finished creating user feature extraction head and loading user dataset")
    
    print (">>> Training user feature extraction model")
    privacy_model_trainer.train()
