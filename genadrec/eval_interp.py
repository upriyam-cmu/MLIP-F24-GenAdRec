# %%
import os
import torch
import numpy as np
import argparse
from dataset.taobao_behavior_dataset_old import TaobaoUserClicksDataset
from model.ad_features_predictor import AdFeaturesPredictor
from torch.optim import AdamW
import torch.nn.functional as F
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

# %%
batch_size = 4804
learning_rate = 1.0
model_dir = os.path.join("models", run_label)

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
print(f"eval only: loading model '{eval_model_id}' from path '{model_dir}'")
model.load_state_dict(torch.load(os.path.join(model_dir, f"{eval_model_id}.pth"), map_location=device)['model_state_dict'])
model.eval()

# %%
embedding_dim_size = model.embedder.output_size
embeddings = torch.zeros(batch_size, embedding_dim_size, device=device, requires_grad=True)
optim = AdamW([embeddings], lr=learning_rate)

# %%
batch_idx = torch.arange(batch_size, device=device)

with tqdm(range(400)) as pbar:
    for it in pbar:
        optim.zero_grad()

        model_output = model.sample(embeddings)[0]  # only consider w.r.t. first feature (ad category)

        target = torch.zeros_like(model_output)
        target[batch_idx, batch_idx] = 1

        loss = F.cross_entropy(model_output, target)

        loss.backward()
        optim.step()

        loss = float(loss)
        grad_norm = float(torch.linalg.vector_norm(embeddings.grad, dim=-1).max())

        pbar.set_postfix({'X-Ent Loss': loss, 'Max Grad Norm': grad_norm, 'lr': learning_rate})

        if (it + 1) % 100 == 0:
            learning_rate /= 10
            optim = AdamW([embeddings], lr=learning_rate)

# %%
def compute_interp_scores(embeddings_np, n_samples):
    latent_wise_corr = np.corrcoef(embeddings_np, rowvar=False)
    corr_score = ((latent_wise_corr - np.eye(embedding_dim_size)) ** 2).mean()

    embd_np_norm = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    similarities_mat = np.einsum('ij,kj->ik', embd_np_norm, embd_np_norm)
    sim_mat_frob = ((similarities_mat - np.eye(n_samples)) ** 2).mean()

    angles = np.acos(np.clip(similarities_mat, -1, 1))
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    assert (angles < 0).sum() == 0
    angle_distr = angles[np.triu_indices(len(angles), k=1)]
    angle_distr_params = float(np.mean(angle_distr)), float(np.std(angle_distr))

    return 1e3 * corr_score, 1e3 * sim_mat_frob, angle_distr_params

embeddings_np = embeddings.detach().cpu().numpy()

uniques, counts = np.unique(train_dataset.ads_data[:, 0], return_counts=True)
category_ids = uniques[np.argsort(counts)][::-1][:15]

all_embeddings_scores = compute_interp_scores(embeddings_np, batch_size)
top_half_embeddings_scores = compute_interp_scores(embeddings_np[category_ids], len(category_ids))

# %%
with open('log.txt', 'a') as log_file:
    def print_and_log(*args, **kwargs):
        print(*args, **kwargs)
        kwargs['file'] = log_file
        print(*args, **kwargs)

    print_and_log(f"eval results for model: type='{run_label}' id='{eval_model_id}'")
    print_and_log("Final Embedding Fit Loss:", loss)

    print_and_log("All Embeddings:")
    corr_score, sim_mat_frob, angle_distr_params = all_embeddings_scores
    print_and_log("\tEmbedding Correlation Matrix Norm:", corr_score)
    print_and_log("\tEmbedding CosineSim Frob Norm:", sim_mat_frob)
    mu, std = angle_distr_params
    print_and_log("\tEmbedding Angle-Delta Distr:", mu, "±", std, "radians")

    print_and_log("Top-Half Categories Embeddings:")
    corr_score, sim_mat_frob, angle_distr_params = top_half_embeddings_scores
    print_and_log("\tEmbedding Correlation Matrix Norm:", corr_score)
    print_and_log("\tEmbedding CosineSim Frob Norm:", sim_mat_frob)
    mu, std = angle_distr_params
    print_and_log("\tEmbedding Angle-Delta Distr:", mu, "±", std, "radians")

    print_and_log()
