# %%
import os
import torch
import numpy as np
import argparse
from dataset.taobao_simple_dataset import TaobaoDataset
from model.ad_features_predictor import AdFeaturesPredictor
from loss.masked_cross_entropy_loss import MaskedCrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# %%
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
device = torch.device(device)
print("Using device:", device)

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--run_label", type=str)
parser.add_argument("--conditional", action="store_true")
parser.add_argument("--residual", action="store_true")
parser.add_argument("--user_feats", action="store_true")

args = parser.parse_args()
run_label = args.run_label
conditional = True          # args.conditional, always conditional
residual = True             # args.residual,    always residual
user_feats = False          # args.user_feats,  pretraining only on user ID models

# %%
batch_size = 1024
learning_rate = 0.001
pretrain_epochs = 1
pretrain_save_eval_every = 2000
model_dir = os.path.join("saved_models", run_label)
outputs_dir = os.path.join("outputs", run_label)

# %%
dataset_params = {
    "data_dir": "data",
    "augmented": True,
    "user_features": ["gender", "age", "shopping", "occupation"] if user_feats else ["user"],
    "ad_features": ["cate", "brand", "customer", "campaign"],
}

# %%
print(">>> Start loading pretraining dataset")
pretrain_dataset = TaobaoDataset(mode="pretrain", **dataset_params)
pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True)
print(">>> Finished loading pretraining dataset")

# %%
print(">>> Start loading test dataset")
test_dataset = TaobaoDataset(mode="test", **dataset_params)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(">>> Finished loading test dataset")

# %%
model = AdFeaturesPredictor(
    input_cardinalities=test_dataset.input_dims,
    embedding_dims=[64] * len(test_dataset.input_dims),
    hidden_dim_specs=[(128, 64)] * len(test_dataset.output_dims),
    output_cardinalities=test_dataset.output_dims,
    residual_connections=residual,
    activation_function='nn.ReLU()',
    device=device,
)
loss_fn = MaskedCrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)

# %%
if not os.path.isdir(outputs_dir):
    os.makedirs(outputs_dir)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

# %%
start_epoch = 0
best_val_loss = float('inf')
train_loss_per_epoch = []
test_loss_per_epoch = []
epoch_train_loss_curve = []

# %%
best_model_path = os.path.join(model_dir, "best_model.pth")
if os.path.isfile(best_model_path):
    print(f">>> Start loading best model: {best_model_path}")
    state_dicts = torch.load(best_model_path)
    start_epoch = state_dicts['epoch'] + 1
    model.load_state_dict(state_dicts['model_state_dict'])
    optimizer.load_state_dict(state_dicts['optimizer_state_dict'])
    best_val_loss = state_dicts['best_val_loss']
    train_loss_per_epoch = state_dicts['train_loss_per_epoch']
    test_loss_per_epoch = state_dicts['test_loss_per_epoch']
    epoch_train_loss_curve = state_dicts['epoch_loss_curves']
    print(f">>> Finished loading best model saved at epoch {start_epoch-1}")

# %%
for epoch in range(pretrain_epochs):
    with tqdm(pretrain_dataloader, desc=f'Pretrain Epoch {epoch}') as pbar:
        pretrain_losses = []
        for user_data, ads_features, ads_masks, _, _ in pbar:
            model.train()
            user_data = user_data.to(device)
            ads_features = ads_features[:, :2].to(device, torch.int64)
            if conditional:
                ads_masks = [None, ads_masks[0].to(device)]
            else:
                ads_masks = [None, None]
            
            optimizer.zero_grad()
            ad_feature_logits = model(user_data)
            loss = loss_fn(
                logits = ad_feature_logits[:2],
                logit_masks = ads_masks, # gen_ads_mask(ads_features, train_dataset, device),
                targets = ads_features
            )
            loss.backward()
            optimizer.step()
            pretrain_losses.append(loss.item())
            pbar.set_postfix({'Loss': loss.item()})
            
            if len(pretrain_losses) % pretrain_save_eval_every == 0:
                torch.save({
                    'epoch': 0,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'train_loss_per_epoch': train_loss_per_epoch,
                    'test_loss_per_epoch': test_loss_per_epoch,
                    'epoch_loss_curves': [pretrain_losses]
                }, os.path.join(model_dir, f'pretrain_{len(pretrain_losses)}.pth'))
                
                model.eval()
                with torch.inference_mode():
                    total_loss = 0
                    with tqdm(test_dataloader) as pbar:
                        batches = 0
                        for user_data, ads_features, ads_masks, _, _ in pbar:
                            user_data = user_data.to(device)
                            ads_features = ads_features[:, :2].to(device, torch.int64)
                            ads_masks = [None, ads_masks[0].to(device)]

                            ad_feature_logits = model(user_data)
                            loss = loss_fn(
                                logits = ad_feature_logits[:2],
                                logit_masks = ads_masks, # gen_ads_mask(ads_features, train_dataset, device),
                                targets = ads_features,
                                penalize_masked = False
                            )
                            total_loss += loss.item()
                            batches += 1
                            pbar.set_postfix({'Loss': total_loss / batches})

                    val_loss = total_loss / batches
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({
                            'epoch': 0,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_val_loss': best_val_loss,
                            'train_loss_per_epoch': train_loss_per_epoch,
                            'test_loss_per_epoch': test_loss_per_epoch,
                            'epoch_loss_curves': [pretrain_losses]
                        }, os.path.join(model_dir, 'best_model.pth'))

        train_loss_per_epoch.append(np.mean(pretrain_losses))
        epoch_train_loss_curve.append(pretrain_losses)