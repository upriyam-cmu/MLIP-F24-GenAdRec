# %%
import os
import torch
import numpy as np
import argparse
from taobao_simple_dataset import TaobaoDataset
from ad_features_predictor import AdFeaturesPredictor
from masked_cross_entropy_loss import MaskedCrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from non_ml_baseline.simple_eval import FrequencyTracker, ReductionTracker, ScoreUtil, compute_ndcg

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
parser.add_argument("--eval_only", action="store_true")
parser.add_argument("--conditional", action="store_true")
parser.add_argument("--residual", action="store_true")
parser.add_argument("--user_feats", action="store_true")

args = parser.parse_args()
run_label = args.run_label
eval_only = args.eval_only
conditional = args.conditional
residual = args.residual
user_feats = args.user_feats

# %%
batch_size = 1024
learning_rate = 0.001
pretrain_epochs = 1
pretrain_save_every = 2000
train_epochs = 30
eval_every_n = 1
save_every_n = 1
model_dir = os.path.join("models", run_label)
outputs_dir = os.path.join("outputs", run_label)

# %%
dataset_params = {
    "data_dir": "../data",
    "min_train_clks": 1,
    "num_test_clks": 1,
    "include_ad_non_clks": False,
    "sequence_mode": False,
    "user_features": ["user", "gender", "age", "shopping", "occupation"] if user_feats else ["user"],
    "ad_features": ["cate", "brand", "customer", "campaign"],
    "conditional_masking": True,
}

# %%
if pretrain_epochs > 0:
    pretrain_dataset = TaobaoDataset(mode="pretrain", **dataset_params)
    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True)

# %%
train_dataset = TaobaoDataset(mode="finetune", **dataset_params)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# %%
if eval_only:
    dataset_params['ad_features'].append("adgroup")
test_dataset = TaobaoDataset(mode="test", **dataset_params)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
if eval_only:
    reduction_tracker = ReductionTracker(test_dataset.ad_features)
    
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pth"))['model_state_dict'])
    model.eval()
    with torch.inference_mode(), tqdm(test_dataloader) as pbar:
        n_users = 0
        ndcg_scores = []
        for user_data, ads_features, _, _, _ in pbar:
            n_users += user_data.shape[0]
            
            user_data = user_data.to(device)
            ads_features = ads_features.to(device)

            ad_feature_logits = *model(user_data), ads_features[:, -1]
            for d_cat, d_brand, d_cust, d_camp, target_ad_id in zip(*ad_feature_logits):
                category_freqs = FrequencyTracker.from_softmax_distribution(d_cat.exp())
                brand_freqs = FrequencyTracker.from_softmax_distribution(d_brand.exp())
                customer_freqs = FrequencyTracker.from_softmax_distribution(d_cust.exp())
                campaign_freqs = FrequencyTracker.from_softmax_distribution(d_camp.exp())

                score_util = ScoreUtil(
                    reduction_tracker=reduction_tracker,
                    category_freqs=category_freqs,
                    brand_freqs=brand_freqs,
                    customer_freqs=customer_freqs,
                    campaign_freqs=campaign_freqs,
                )
                
                ndcg_scores.append(compute_ndcg(
                    score_util,
                    target_ad_id=target_ad_id.item(),
                    subsampling=1 / 20,
                    verbose=False,
                    use_tqdm=True,
                ))

                if len(ndcg_scores) > 60:
                    print(run_label, np.mean(ndcg_scores))
                    exit(0)


# %%
loss_fn = MaskedCrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)

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

# %%
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
else:
    for epoch in range(pretrain_epochs):
        model.train()
        with tqdm(pretrain_dataloader, desc=f'Pretrain Epoch {epoch}') as pbar:
            pretrain_losses = []
            for user_data, ads_features, ads_masks, _, _ in pbar:
                user_data = user_data.to(device)
                ads_features = ads_features[:, :2].to(device)
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
                
                if len(pretrain_losses) % pretrain_save_every == 0:
                    torch.save({
                        'epoch': 0,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'train_loss_per_epoch': train_loss_per_epoch,
                        'test_loss_per_epoch': test_loss_per_epoch,
                        'epoch_loss_curves': [pretrain_losses]
                    }, os.path.join(model_dir, f'pretrain_{len(pretrain_losses)}.pth'))

            train_loss_per_epoch.append(np.mean(pretrain_losses))
            epoch_train_loss_curve.append(pretrain_losses)


# %%
for epoch in range(start_epoch, train_epochs):
    model.train()
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
            ad_feature_logits = model(user_data)
            loss = loss_fn(
                logits = ad_feature_logits,
                logit_masks = ads_masks, # gen_ads_mask(ads_features, train_dataset, device),
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
                    ads_masks = [None] + [mask.to(device) for mask in ads_masks]

                    ad_feature_logits = model(user_data)
                    loss = loss_fn(
                        logits = ad_feature_logits,
                        logit_masks = ads_masks, # gen_ads_mask(ads_features, train_dataset, device),
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

    # np.save(os.path.join(outputs_dir, 'train_loss_per_epoch.npy'), train_loss_per_epoch)
    # np.save(os.path.join(outputs_dir, 'test_loss_per_epoch.npy'), test_loss_per_epoch)
    # np.save(os.path.join(outputs_dir, 'epoch_loss_curves.npy'), epoch_train_loss_curve)
