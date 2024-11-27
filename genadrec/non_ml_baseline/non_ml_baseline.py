import numpy as np
from tqdm import tqdm
from sys import stderr

from simple_eval import FrequencyTracker, ReductionTracker, ScoreUtil, compute_ndcg


# data format schema
## user data -- not used
## ad data -- [ad_id, category, brand, customer, campaign]
## behavior data -- [user_id, time_stamp, category, brand]
## interaction data -- [user_id, time_stamp, ad_id, click]


reduction_tracker = None

def run_baseline(user_data, ad_data, behavior_data, interaction_data, *, sample_idx=None, sample_last=True, verbose=False, use_tqdm=True):
    # drop non-click interactions from dataset
    interaction_data = interaction_data[interaction_data[:, 3] == 1]

    # select target click event
    if not sample_last:
        # sample any interaction split
        n_interactions = len(interaction_data)
        if sample_idx is None:
            sample_idx = np.random.choice(n_interactions)
        elif not (0 <= sample_idx < n_interactions):
            print(f"Expected sample_idx in [0,{n_interactions}). Got {sample_idx}. Projecting using modulo.", file=stderr)
            sample_idx = (sample_idx % n_interactions + n_interactions) % n_interactions
    else:
        # sample last interaction split for user
        unique_users, interaction_inds = np.unique(interaction_data[::-1, 0], return_index=True)
        interaction_inds = len(interaction_data) - interaction_inds - 1
        assert np.allclose(interaction_data[interaction_inds, 0], unique_users)

        n_interactions = len(interaction_inds)
        if sample_idx is None:
            sample_idx = np.random.choice(interaction_inds)
        elif not (0 <= sample_idx < n_interactions):
            print(f"Expected sample_idx in [0,{n_interactions}). Got {sample_idx}. Projecting using modulo.", file=stderr)
            sample_idx = (sample_idx % n_interactions + n_interactions) % n_interactions
            sample_idx = interaction_inds[sample_idx]

    # identify interaction data
    i_user_id, i_time_stamp, i_ad_id, i_click = interaction_data[sample_idx]
    assert i_click == 1

    # filter remaining dataset by user & timestamp
    behavior_data = behavior_data[(behavior_data[:, 0] == i_user_id) & (behavior_data[:, 1] <= i_time_stamp)]
    interaction_data = np.delete(interaction_data, (sample_idx,), axis=0)  # drop target sample first
    interaction_data = interaction_data[(interaction_data[:, 0] == i_user_id) & (interaction_data[:, 1] <= i_time_stamp)]
    assert i_ad_id not in interaction_data[(interaction_data[:, 1] == i_time_stamp), 2]  # can't have target interaction in history

    # identify ad data from ads user previously interacted with
    interacted_ad_data = ad_data[np.isin(ad_data[:, 0], interaction_data[:, 2])]

    # generate frequency tables & util for scoring
    category_freqs = FrequencyTracker(interacted_ad_data[:, 1], behavior_data[:, 2]).count(ad_data[:, 1], weight=1.0)
    brand_freqs = FrequencyTracker(interacted_ad_data[:, 2], behavior_data[:, 3]).count(ad_data[:, 2], weight=1.0)
    customer_freqs = FrequencyTracker(interacted_ad_data[:, 3]).count(ad_data[:, 3], weight=1.0)
    campaign_freqs = FrequencyTracker(interacted_ad_data[:, 4]).count(ad_data[:, 4], weight=1.0)

    global reduction_tracker
    if reduction_tracker is None:
        reduction_tracker = ReductionTracker(ad_data)

    score_util = ScoreUtil(
        reduction_tracker=reduction_tracker,
        category_freqs=category_freqs,
        brand_freqs=brand_freqs,
        customer_freqs=customer_freqs,
        campaign_freqs=campaign_freqs,
    )

    # compute & return ndcg score
    return compute_ndcg(
        score_util,
        target_ad_id=i_ad_id,
        subsampling=1 / 20,
        verbose=verbose,
        use_tqdm=use_tqdm,
    )


if __name__ == '__main__':
    import pandas as pd

    user_data = pd.read_csv('./dataset/user_data.csv')
    ad_data = pd.read_csv('./dataset/ad_data.csv')
    behavior_data = pd.read_csv('./dataset/behavior_data.csv')
    interaction_data = pd.read_csv('./dataset/interaction_data.csv')

    from preprocessing import preprocess

    processed_data = preprocess(
        user_data=user_data,
        ad_data=ad_data,
        behavior_data=behavior_data,
        interaction_data=interaction_data,
        inplace=False,
        time_stamp_block_size=900,
    )

    ndcg_scores = []
    with tqdm(range(1)) as pbar:
        for _ in pbar:
            pbar.set_postfix(avg_ndcg=np.mean(ndcg_scores))
            ndcg_score = run_baseline(*processed_data)
            ndcg_scores.append(ndcg_score)
    print("ndcg scores:", ndcg_scores)
    print("avg ndcg:", np.mean(ndcg_scores))
