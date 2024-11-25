# %%
import os
import torch
import numpy as np
import argparse
from taobao_simple_dataset import TaobaoDataset
from ad_features_predictor import AdFeaturesPredictor
from masked_cross_entropy_loss import MaskedCrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from non_ml_baseline.simple_eval import OptimizedFrequencyTracker as FrequencyTracker, ReductionTracker, ScoreUtil, compute_ndcg

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
parser.add_argument("--run_label", type=str, default="")
parser.add_argument("--eval_model_id", type=str)
parser.add_argument("--conditional", action="store_true")
parser.add_argument("--residual", action="store_true")
parser.add_argument("--user_feats", action="store_true")

args = parser.parse_args()
run_label = args.run_label
eval_model_id = args.eval_model_id
conditional = args.conditional
residual = args.residual
user_feats = args.user_feats

if run_label == "":
    run_label = '-'.join([
        'conditional' if conditional else 'marginal',
        'residual' if residual else 'independent',
        'user_fts' if user_feats else 'uid_only',
    ])

# %%
batch_size = 1024
model_dir = os.path.join("models", run_label)

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
loss_eval_dataset = TaobaoDataset(mode="test", **dataset_params)
validation_loader = DataLoader(loss_eval_dataset, batch_size=batch_size, shuffle=True)

# %%
dataset_params = {
    **dataset_params,
    "ad_features": ["adgroup"] + dataset_params["ad_features"],
}
test_dataset = TaobaoDataset(mode="test", **dataset_params)
ndcg_test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
model = AdFeaturesPredictor(
    input_cardinalities=loss_eval_dataset.input_dims,
    embedding_dims=[64] * len(loss_eval_dataset.input_dims),
    hidden_dim_specs=[(128, 64)] * len(loss_eval_dataset.output_dims),
    output_cardinalities=loss_eval_dataset.output_dims,
    residual_connections=residual,
    activation_function='nn.ReLU()',
    device=device,
)

# %%
ad_features = test_dataset.ad_features
ad_features = ad_features[ad_features[:, 0] != -1]
reduction_tracker = ReductionTracker(ad_features)

print(f"eval only: loading model '{eval_model_id}' from path '{model_dir}'")
model.load_state_dict(torch.load(os.path.join(model_dir, f"{eval_model_id}.pth"))['model_state_dict'])
model.eval()

sample_limit = 120
loss_fn = MaskedCrossEntropyLoss()

with torch.inference_mode():
    with tqdm(validation_loader) as pbar:
        total_loss, batches = 0, 0

        for user_data, ads_features, ads_masks, _, _ in pbar:
            user_data = user_data.to(device)
            ads_features = ads_features.to(device)
            if conditional:
                ads_masks = [None] + [mask.to(device) for mask in ads_masks]
            else:
                ads_masks = [None] * (len(ads_masks) + 1)
            
            ad_feature_logits = model(user_data)
            loss = loss_fn(
                logits = ad_feature_logits,
                logit_masks = ads_masks,
                targets = ads_features,
                penalize_masked = False,
            )
            total_loss += float(loss)
            batches += 1
            pbar.set_postfix({'Loss': total_loss / batches})

        validation_loss = total_loss / batches

    def iterate_samples():
        for user_data, ads_features, ads_masks, _, _ in ndcg_test_dataloader:            
            user_data = user_data.to(device)
            ads_features = ads_features.to(device)
            if conditional:
                ads_masks = [None] + [mask.to(device) for mask in ads_masks]
            else:
                ads_masks = [None] * (len(ads_masks) + 1)

            ad_feature_logits = *model(user_data), ads_features[:, 0]
            for sample in zip(*ad_feature_logits):
                yield sample

    with tqdm(range(sample_limit)) as pbar:
        it = iter(iterate_samples())
        ndcg_scores = []

        for _ in pbar:
            d_cat, d_brand, d_cust, d_camp, target_ad_id = next(it)

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
                subsampling=0.1,
                verbose=False,
                use_tqdm=False,
            ))

            pbar.set_postfix({'avg NDCG': float(np.mean(ndcg_scores))})

with open('log.txt', 'a') as log_file:
    def print_and_log(*args, **kwargs):
        print(*args, **kwargs)
        kwargs['file'] = log_file
        print(*args, **kwargs)

    print_and_log(f"eval results for model: type='{run_label}' id='{eval_model_id}'")
    print_and_log("Validation Loss:", validation_loss)
    print_and_log("Avg NDCG:", float(np.mean(ndcg_scores)))
    print_and_log()