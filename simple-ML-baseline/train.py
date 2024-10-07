# %%
import torch
from taobao_behavior_dataset import TaobaoUserClicksDataset
from ad_features_predictor import AdFeaturesPredictor
from masked_cross_entropy_loss import MaskedCrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
batch_size = 32
learning_rate = 0.001
train_epochs = 1000
eval_every_n = 10

# %%
dataset_params = {
    "data_dir": "data",
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
    ads_masks = [torch.ones((len(ads_data), dim), dtype=bool, device=device) for dim in dataset.output_dims[1:]]
    for i in range(len(ads_data)):
        ad_data = ads_data[i]
        if dataset.include_ad_ids:
            ad_data = ad_data[1:]
        for j, mask in enumerate(ads_masks):
            mask[i, dataset.conditional_mappings[j][tuple(ad_data[:j+1].tolist())]] = False
    return [None] + ads_masks

# %%
for epoch in range(train_epochs):
    model.train()
    with tqdm(train_dataloader, desc=f'Epoch {epoch+1}') as pbar:
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
            pbar.set_postfix({'Loss': loss.item()})

    if epoch % eval_every_n == 0:
        model.eval()
        with torch.no_grad():
            with tqdm(test_dataloader) as pbar:
                for user_data, ads_features, _, _ in pbar:
                    user_data = user_data.to(device)
                    ads_features = ads_features.to(device)
                    logits = model(user_data)
                    loss = loss_fn(
                        logits = ad_feature_logits,
                        logit_masks = gen_ads_mask(ads_features, train_dataset, device),
                        targets = ads_features
                    )
                    pbar.set_postfix({'Loss': loss.item()})
