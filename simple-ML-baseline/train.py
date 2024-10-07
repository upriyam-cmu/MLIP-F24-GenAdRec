# %%
import torch
import numpy as np
from taobao_behavior_dataset import TaobaoUserClicksDataset
from ad_features_predictor import AdFeaturesPredictor
from masked_cross_entropy_loss import MaskedCrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %%
batch_size = 32
learning_rate = 0.001
train_epochs = 2
eval_every_n = 10

# %%
dataset_params = {
    "data_dir": "../data",
    "filter_clicks": True,
    "include_user_ids": True,
    "user_features": [],
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
train_loss_per_epoch = []
epoch_train_loss_curve = []
test_loss_per_epoch = []

best_val_loss = float('inf')
for epoch in range(train_epochs):
    model.train()
    with tqdm(train_dataloader, desc=f'Epoch {epoch+1}') as pbar:
        train_losses = []
        for user_data, ads_features, _, _ in pbar:
            user_data = user_data.to(device)
            ads_features = ads_features.to(device)
            
            optimizer.zero_grad()
            ad_feature_logits = model(user_data)
            loss = loss_fn(
                logits = ad_feature_logits,
                logit_masks = gen_ads_mask(ads_features, train_dataset, device),
                targets = ads_features
            )
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix({'Loss': np.mean(train_losses)})
        train_loss_per_epoch.append(np.mean(train_losses))
        epoch_train_loss_curve.append(train_losses)

    if epoch % eval_every_n == 0:
        model.eval()
        with torch.no_grad():
            total_loss = 0
            with tqdm(test_dataloader) as pbar:
                batches = 0
                for user_data, ads_features, _, _ in pbar:
                    user_data = user_data.to(device)
                    ads_features = ads_features.to(device)
                    ad_feature_logits = model(user_data)
                    loss = loss_fn(
                        logits = ad_feature_logits,
                        logit_masks = gen_ads_mask(ads_features, train_dataset, device),
                        targets = ads_features,
                        penalize_masked = False
                    )
                    total_loss += loss.item()
                    batches += 1
                    pbar.set_postfix({'Loss': total_loss / batches})

            val_loss = total_loss / len(test_dataloader)
            test_loss_per_epoch.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, f'models/best_model_{epoch}.pth')
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, f'models/epoch_{epoch}.pth')

# %%
np.save('outputs/train_loss_per_epoch.npy', train_loss_per_epoch)
np.save('outputs/test_loss_per_epoch.npy', test_loss_per_epoch)
np.save('outputs/loss_curves.npy', epoch_train_loss_curve)
