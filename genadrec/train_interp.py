# %%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from dataset.taobao_behavior_dataset import TaobaoUserClicksDataset
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
parser.add_argument("--discriminator_loss_weight", type=float, default=0)
parser.add_argument("--loopback_loss_weight", type=float, default=0)

args = parser.parse_args()
run_label = args.run_label
discriminator_loss_weight = args.discriminator_loss_weight
loopback_loss_weight = args.loopback_loss_weight

conditional = True
residual = False
user_feats = False
use_discriminator_loss = discriminator_loss_weight > 0
use_loopback_loss = loopback_loss_weight > 0

# %%
batch_size = 384
learning_rate = 0.001
train_epochs = 30
eval_every_n = 1
save_every_n = 1
model_dir = os.path.join("models", run_label)
outputs_dir = os.path.join("outputs", run_label)

# %%
dataset_params = {
    "data_dir": "raw_data",
    "filter_clicks": True,
    "include_user_ids": True,
    "user_features": ["final_gender_code", "age_level", "shopping_level", "occupation"] if user_feats else [],
    "include_ad_ids": False,
    "ad_features": ["cate_id", "brand", "customer", "campaign_id"],
}
train_dataset = TaobaoUserClicksDataset(training=True, **dataset_params)
test_dataset = TaobaoUserClicksDataset(training=False, **dataset_params)

# %%
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# %%
model = AdFeaturesPredictor(
    input_cardinalities=train_dataset.input_dims,
    embedding_dims=[64] * len(train_dataset.input_dims),
    hidden_dim_specs=[(128, 64)] * len(train_dataset.output_dims),
    output_cardinalities=train_dataset.output_dims,
    residual_connections=residual,
    activation_function='nn.ReLU()',
    device=device,
)

# %%
# aux loss models
if use_discriminator_loss:
    embed_size = model.embedder.output_size
    discriminator = nn.Sequential(
        nn.Linear(embed_size, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )
    discriminator.to(device)
    discriminator_optimizer = AdamW(discriminator.parameters(), lr=learning_rate, maximize=True)

if use_loopback_loss:
    distr_dim_size = sum(train_dataset.output_dims)
    embed_size = model.embedder.output_size
    loopback_model = nn.Sequential(
        nn.Linear(distr_dim_size, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    )
    loopback_model.to(device)

    embedding_projector = nn.Sequential(
        nn.Linear(batch_size, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
    )
    embedding_projector.to(device)

    loopback_model_optimizer = AdamW(
        list(loopback_model.parameters()) + list(embedding_projector.parameters()),
        lr=learning_rate,
    )

# %%
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
test_loss_per_epoch = []
epoch_train_loss_curve = []

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
        train_loss_per_epoch = state_dicts['train_loss_per_epoch']
        test_loss_per_epoch = state_dicts['test_loss_per_epoch']
        epoch_train_loss_curve = state_dicts['epoch_loss_curves']

# %%
def compute_loss(user_data, ads_features, ads_masks, epoch=None):
    user_embeddings = model.embed(user_data)
    ad_feature_logits = model.sample(user_embeddings)
    loss = loss_fn(
        logits=ad_feature_logits,
        logit_masks=ads_masks, # gen_ads_mask(ads_features, train_dataset, device),
        targets=ads_features,
        penalize_masked=False,
    )

    keep_fakes = epoch is None or epoch > 0.15

    if keep_fakes and (use_discriminator_loss or use_loopback_loss):
        b_sz = len(user_data)
        batch_inds = torch.arange(b_sz).to(device)
        modified_indices = torch.tensor(np.random.choice(user_embeddings.shape[-1], size=b_sz), dtype=int).to(device)
        replacement_inds = np.random.choice(b_sz - 1, size=b_sz)
        replacement_inds += (replacement_inds >= np.arange(b_sz))
        replacement_inds = torch.tensor(replacement_inds, dtype=int).to(device)
        modified_user_embeddings = user_embeddings.detach().clone()
        modified_user_embeddings[batch_inds, modified_indices] = modified_user_embeddings[replacement_inds, modified_indices]

    discriminator_gain = 0
    if use_discriminator_loss:
        trues_loss = (1 - discriminator(user_embeddings)).log().mean()
        if keep_fakes:
            fakes_loss = discriminator(modified_user_embeddings).log().mean()
        else:
            fakes_loss = 0
        discriminator_gain = trues_loss + fakes_loss

    loopback_loss = 0
    if keep_fakes and use_loopback_loss:
        modified_logits = model.sample(modified_user_embeddings)
        modified_logits = [F.softmax(logits, dim=-1) for logits in modified_logits]
        modified_logits = torch.cat(modified_logits, dim=-1)

        pred_embed_weights = loopback_model(modified_logits)  # (N, x?)

        u_embed_T = torch.transpose(user_embeddings, -1, -2)  # (F, N)
        u_proj_T = embedding_projector(u_embed_T)  # (F, x?)
        user_embed_proj = torch.transpose(u_proj_T, -1, -2)  # (x?, F)

        pred_modified_latents = pred_embed_weights @ user_embed_proj

        target_values = modified_user_embeddings[batch_inds, modified_indices]
        predicted_values = pred_modified_latents[batch_inds, modified_indices]

        loopback_loss = ((target_values - predicted_values) ** 2).mean()

    total_loss = loss + discriminator_loss_weight * discriminator_gain + loopback_loss_weight * loopback_loss
    return total_loss, loss, discriminator_gain, loopback_loss

# %%
for epoch in range(start_epoch, train_epochs):
    model.train()
    if use_discriminator_loss:
        discriminator.train()
    if use_loopback_loss:
        loopback_model.train()
        embedding_projector.train()

    with tqdm(train_dataloader, desc=f'Epoch {epoch}') as pbar:
        train_losses = []
        for user_data, ads_features, ads_masks, _, _ in pbar:
            user_data = user_data.to(device)
            ads_features = ads_features.to(device)
            if conditional:
                ads_masks = [None] + [mask.to(device) for mask in ads_masks]
            else:
                ads_masks = [None] * (len(ads_masks)+1)
            
            optimizer.zero_grad()
            if use_discriminator_loss:
                discriminator_optimizer.zero_grad()
            if use_loopback_loss:
                loopback_model_optimizer.zero_grad()

            total_loss, model_loss, gan_loss, loopback_loss = compute_loss(
                user_data=user_data,
                ads_features=ads_features,
                ads_masks=ads_masks,
                epoch=(epoch + pbar.n / pbar.total),  # real-valued progress
            )
            total_loss.backward()

            optimizer.step()
            if use_discriminator_loss:
                discriminator_optimizer.step()
            if use_loopback_loss:
                loopback_model_optimizer.step()

            # bookkeeping
            train_losses.append(total_loss.item())
            pbar.set_postfix({
                'Total Loss': float(total_loss),
                'Model Loss': float(model_loss),
                'GAN Loss': float(gan_loss),
                'Loopback Loss': float(loopback_loss),
            })
        train_loss_per_epoch.append(float(np.mean(train_losses)))
        epoch_train_loss_curve.append(train_losses)

    if epoch % eval_every_n == 0:
        model.eval()
        if use_discriminator_loss:
            discriminator.eval()
        if use_loopback_loss:
            loopback_model.eval()
            embedding_projector.eval()

        with torch.inference_mode():
            losses = np.array([0.] * 4)
            with tqdm(test_dataloader, desc='EVAL') as pbar:
                batches = 0
                for user_data, ads_features, ads_masks, _, _ in pbar:
                    user_data = user_data.to(device)
                    ads_features = ads_features.to(device)
                    if conditional:
                        ads_masks = [None] + [mask.to(device) for mask in ads_masks]
                    else:
                        ads_masks = [None] * (len(ads_masks)+1)

                    # ad_feature_logits = model(user_data)
                    # loss = loss_fn(
                    #     logits = ad_feature_logits,
                    #     logit_masks = ads_masks, # gen_ads_mask(ads_features, train_dataset, device),
                    #     targets = ads_features
                    # )
                    loss, model_loss, gan_loss, loopback_loss = compute_loss(user_data=user_data, ads_features=ads_features, ads_masks=ads_masks)

                    losses += np.array([
                        float(loss),
                        float(model_loss),
                        float(gan_loss),
                        float(loopback_loss),
                    ])
                    batches += 1
                    
                    val_loss = losses / batches
                    pbar.set_postfix(dict(zip(('Total Loss', 'Model Loss', 'GAN Loss', 'Loopback Loss'), val_loss.tolist())))
            
            test_loss_per_epoch.append(val_loss.tolist())

            if val_loss[1] < best_val_loss:
                best_val_loss = val_loss[1]
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'train_loss_per_epoch': train_loss_per_epoch,
                    'test_loss_per_epoch': test_loss_per_epoch,
                    'epoch_loss_curves': epoch_train_loss_curve
                }, os.path.join(model_dir, 'best_model.pth'))

    if epoch % save_every_n == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'train_loss_per_epoch': train_loss_per_epoch,
            'test_loss_per_epoch': test_loss_per_epoch,
            'epoch_loss_curves': epoch_train_loss_curve
        }, os.path.join(model_dir, f'epoch_{epoch}.pth'))

    np.save(os.path.join(outputs_dir, 'train_loss_per_epoch.npy'), train_loss_per_epoch)
    np.save(os.path.join(outputs_dir, 'test_loss_per_epoch.npy'), test_loss_per_epoch)
    np.save(os.path.join(outputs_dir, 'epoch_loss_curves.npy'), epoch_train_loss_curve)
