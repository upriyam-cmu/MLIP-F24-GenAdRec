import numpy as np
import torch
import torch.nn as nn

from non_ml_baseline.simple_eval import FrequencyTracker, ReductionTracker, ScoreUtil, compute_ndcg


# user_features: (N, F) > N = batch/sample size
# target_ad_id: (N,)
# ad_data : (D, 5) > [ad_id, category, brand, customer, campaign]
def score_model_ndcg(model: nn.Module, dataset, *, device, batch_size: int = 128):
    reduction_tracker = ReductionTracker(dataset.ad_features)

    user_features = torch.Tensor(dataset.u, device=device)
    batches = torch.split(user_features, batch_size, 0)

    ndcg_scores = []
    n_users = 0
    for batch in batches:
        print(f"Processing users {n_users} to {n_users + len(batch) - 1}")
        n_users += len(batch)

        for d_cat, d_brand, d_cust, d_camp in zip(*model(batch)):
            # generate frequency tables & util for scoring
            category_freqs = FrequencyTracker.from_softmax_distribution(d_cat.cpu().numpy().exp())
            brand_freqs = FrequencyTracker.from_softmax_distribution(d_brand.cpu().numpy().exp())
            customer_freqs = FrequencyTracker.from_softmax_distribution(d_cust.cpu().numpy().exp())
            campaign_freqs = FrequencyTracker.from_softmax_distribution(d_camp.cpu().numpy().exp())

            score_util = ScoreUtil(
                reduction_tracker=reduction_tracker,
                category_freqs=category_freqs,
                brand_freqs=brand_freqs,
                customer_freqs=customer_freqs,
                campaign_freqs=campaign_freqs,
            )

            # compute ndcg score
            ndcg_scores.append(compute_ndcg(
                score_util,
                target_ad_id=target_ad_id,
                subsampling=1 / 20,
                verbose=False,
                use_tqdm=True,
            ))

    # return avg ndcg
    return np.mean(ndcg_scores)
