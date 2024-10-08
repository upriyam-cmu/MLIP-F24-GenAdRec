# %%
import os
import torch
import numpy as np
import argparse
from taobao_behavior_dataset import TaobaoUserClicksDataset
from ad_features_predictor import AdFeaturesPredictor
from masked_cross_entropy_loss import MaskedCrossEntropyLoss
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
parser.add_argument("--user_feats", action="store_true")

args = parser.parse_args()
run_label = args.run_label
conditional = args.conditional
user_feats = args.user_feats

# %%
batch_size = 2048
learning_rate = 0.001
train_epochs = 20
eval_every_n = 1
save_every_n = 1
model_dir = os.path.join("models", run_label)
outputs_dir = os.path.join("outputs", run_label)

# %%
dataset_params = {
    "data_dir": "../data",
    "filter_clicks": True,
    "include_user_ids": True,
    "user_features": ["final_gender_code", "age_level", "shopping_level", "occupation"] if user_feats else [],
    "include_ad_ids": False,
    "ad_features": ["cate_id", "brand", "customer", "campaign_id"],
}
train_dataset = TaobaoUserClicksDataset(training=True, **dataset_params)
test_dataset = TaobaoUserClicksDataset(training=False, **dataset_params)

# %%
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
model = AdFeaturesPredictor(
    input_cardinalities=train_dataset.input_dims,
    embedding_dims=[64] * len(train_dataset.input_dims),
    hidden_dim_specs=[(128, 64)] * len(train_dataset.output_dims),
    output_cardinalities=train_dataset.output_dims,
    conditioned=conditional,
    activation_function='nn.ReLU()',
    device=device,
)
loss_fn = MaskedCrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)

# %%
def gen_ads_mask(ads_data, dataset, device):
    ads_masks = [np.ones((len(ads_data), dim), dtype=bool) for dim in dataset.output_dims[1:]]
    for i in range(len(ads_data)):
        ad_data = ads_data[i]
        if dataset.include_ad_ids:
            ad_data = ad_data[1:]
        for j, mask in enumerate(ads_masks):
            mask[i, dataset.conditional_mappings[j][tuple(ad_data[:j+1].tolist())]] = False
    ads_masks = [torch.tensor(mask, dtype=torch.bool, device=device) for mask in ads_masks]
    return [None] + ads_masks

# %%
if not os.path.isdir(outputs_dir):
    os.makedirs(outputs_dir)

# %%
start_epoch = 0
best_val_loss = float('inf')
train_loss_per_epoch = []
epoch_train_loss_curve = []
test_loss_per_epoch = []

# %%
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
else:
    best_model_path = os.path.join(model_dir, "best_model.pth")
    if os.path.isfile(best_model_path):
        state_dicts = torch.load(best_model_path)
        start_epoch = state_dicts['epoch'] + 1
        model.load_state_dict(state_dicts['model_state_dict'])
        optimizer.load_state_dict(state_dicts['optimizer_state_dict'])
        best_val_loss = state_dicts['best_val_loss']

# %%
for epoch in range(start_epoch, train_epochs):
    model.train()
    with tqdm(train_dataloader, desc=f'Epoch {epoch}') as pbar:
        train_losses = []
        for user_data, ads_features, ads_masks, _, _ in pbar:
            user_data = user_data.to(device)
            ads_features = ads_features.to(device)
            ads_masks = [mask.to(device) for mask in ads_masks]
            
            optimizer.zero_grad()
            ad_feature_logits = model(user_data)
            loss = loss_fn(
                logits = ad_feature_logits,
                logit_masks = [None] + ads_masks, # gen_ads_mask(ads_features, train_dataset, device),
                targets = ads_features
            )
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix({'Loss': loss.item()})
        train_loss_per_epoch.append(np.mean(train_losses))
        epoch_train_loss_curve.append(train_losses)

    if epoch % eval_every_n == 0:
        model.eval()
        with torch.no_grad():
            total_loss = 0
            with tqdm(test_dataloader) as pbar:
                batches = 0
                for user_data, ads_features, ads_masks, _, _ in pbar:
                    user_data = user_data.to(device)
                    ads_features = ads_features.to(device)
                    ads_masks = [mask.to(device) for mask in ads_masks]

                    ad_feature_logits = model(user_data)
                    loss = loss_fn(
                        logits = ad_feature_logits,
                        logit_masks = [None] + ads_masks, # gen_ads_mask(ads_features, train_dataset, device),
                        targets = ads_features,
                        penalize_masked = False
                    )
                    total_loss += loss.item()
                    batches += 1
                    pbar.set_postfix({'Loss': total_loss / batches})

            val_loss = total_loss / batches
            test_loss_per_epoch.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }, os.path.join(model_dir, 'best_model.pth'))

    if epoch % save_every_n == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, os.path.join(model_dir, f'epoch_{epoch}.pth'))

    np.save(os.path.join(outputs_dir, 'train_loss_per_epoch.npy'), train_loss_per_epoch)
    np.save(os.path.join(outputs_dir, 'test_loss_per_epoch.npy'), test_loss_per_epoch)
    np.save(os.path.join(outputs_dir, 'epoch_loss_curves.npy'), epoch_train_loss_curve)
