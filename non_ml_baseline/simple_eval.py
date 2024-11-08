import numpy as np
from tqdm import tqdm
from typing import Union


class FrequencyTracker:
    def __init__(self, *data: np.ndarray):
        self._freq = {}
        self._tot_count = 0

        for arr in data:
            self.count(arr)

    def count(self, arr: np.ndarray, *, weight=None):
        values, counts = np.unique(arr.flatten(), return_counts=True)

        if weight is not None:
            counts = weight * counts / counts.sum()

        for v, c in zip(values, counts):
            if v not in self._freq:
                self._freq[v] = 0
            self._freq[v] += c
            self._tot_count += c

        return self

    def p_normalized(self, candidates, /):
        freqs = np.array([self._freq.get(v, 0) for v in candidates.flatten()]) + 1e-15
        return freqs / freqs.sum()

    @staticmethod
    def from_softmax_distribution(distribution: np.ndarray, /) -> 'FrequencyTracker':
        ret = FrequencyTracker()
        for idx, freq in enumerate(distribution.flatten()):
            ret._freq[idx] = freq.item()
        ret._tot_count = distribution.sum().item()
        return ret


class ReductionTracker:
    def __init__(self, all_ad_data):
        self._inter_cache, self._ret_cache = {}, {}

        # [ad_id, category, brand, customer, campaign]
        self.ad_data = all_ad_data
        assert self.ad_data.shape[-1] == 5

    def _reduce(self, *keys) -> np.ndarray:
        if keys not in self._ret_cache:
            ad_data = self.ad_data[:, 1:]
            key_trace = ()
            for key in keys:
                key_trace = *key_trace, key
                if key_trace not in self._inter_cache:
                    ad_data = ad_data[(ad_data[:, 0] == key), 1:]
                    self._inter_cache[key_trace] = ad_data
                else:
                    ad_data = self._inter_cache[key_trace]
            self._ret_cache[keys] = np.unique(ad_data[:, 0])
        return self._ret_cache[keys]

    def reduce_categories(self):
        return self._reduce()

    def reduce_brands(self, category):
        return self._reduce(category)
    
    def reduce_customers(self, category, brand):
        return self._reduce(category, brand)
    
    def reduce_campaigns(self, category, brand, customer):
        return self._reduce(category, brand, customer)


class ScoreUtil:
    reduction_tracker: ReductionTracker

    category_freqs: FrequencyTracker
    brand_freqs: FrequencyTracker
    customer_freqs: FrequencyTracker
    campaign_freqs: FrequencyTracker

    def __init__(self, reduction_tracker, category_freqs, brand_freqs, customer_freqs, campaign_freqs):
        self.reduction_tracker = reduction_tracker

        self.category_freqs = category_freqs
        self.brand_freqs = brand_freqs
        self.customer_freqs = customer_freqs
        self.campaign_freqs = campaign_freqs

    @staticmethod
    def _score_feature(freq_tracker: FrequencyTracker, all_options: np.ndarray, chosen_option: int):
        idx = np.searchsorted(all_options, chosen_option)
        assert all_options[idx] == chosen_option
        freqs = freq_tracker.p_normalized(all_options)
        assert freqs.shape == all_options.shape
        return freqs[idx]

    def score_ad(self, category: int, brand: int, customer: int, campaign: int):
        # score category feature
        categories = self.reduction_tracker.reduce_categories()
        category_score = self._score_feature(self.category_freqs, categories, category)

        # score brand feature
        brands = self.reduction_tracker.reduce_brands(category=category)
        brand_score = self._score_feature(self.brand_freqs, brands, brand)

        # score customer feature
        customers = self.reduction_tracker.reduce_customers(category=category, brand=brand)
        customer_score = self._score_feature(self.customer_freqs, customers, customer)

        # score campaign feature
        campaigns = self.reduction_tracker.reduce_campaigns(category=category, brand=brand, customer=customer)
        campaign_score = self._score_feature(self.campaign_freqs, campaigns, campaign)

        # combine scores
        raw_scores = category_score, brand_score, customer_score, campaign_score
        return np.prod(raw_scores), raw_scores


def compute_ndcg(
        score_util: ScoreUtil, /, *,
        target_ad_id: int,
        subsampling: Union[int, float, None] = None,
        verbose: bool = False,
        use_tqdm: bool = True,
) -> float:
    ad_data = np.copy(score_util.reduction_tracker.ad_data)
    np.random.shuffle(ad_data)

    target_ad_data = ad_data[ad_data[:, 0] == target_ad_id]
    assert target_ad_data.shape == (1, 5), f"{target_ad_data.shape} != (1, 5), {ad_data.shape}, {target_ad_id}"
    ad_id, category, brand, customer, campaign = target_ad_data.flatten()
    assert ad_id == target_ad_id
    target_ad_score, feature_scores = score_util.score_ad(category, brand, customer, campaign)
    if verbose:
        print("ad score:", target_ad_score)
        print("feature scores: (category, brand, customer, campaign)", feature_scores)
        print()

    # score all ads
    if subsampling is None:
        subsampling = 1.0
    if isinstance(subsampling, float):
        subsampling = int(round(len(ad_data) * subsampling))
    if isinstance(subsampling, int):
        subsampling = int(np.clip(subsampling, 0, len(ad_data)))
    assert isinstance(subsampling, int) and 0 < subsampling < len(ad_data)

    random_subsample = ad_data[:subsampling]
    if use_tqdm:
        random_subsample = tqdm(random_subsample)
    all_ad_scores = np.array([score_util.score_ad(*ad_features)[0] for _, *ad_features in random_subsample])
    if verbose:
        print("max score:", all_ad_scores.max())
        print("min score:", all_ad_scores.min())
        print("avg score:", all_ad_scores.mean())
        print()

    low_rank, high_rank = (all_ad_scores > target_ad_score).sum() + 0.5, (all_ad_scores >= target_ad_score).sum() + 0.5
    low_q, high_q = low_rank / len(all_ad_scores), high_rank / len(all_ad_scores)
    avg_q = (low_q + high_q) / 2
    est_rank = avg_q * len(ad_data)
    if verbose:
        print("computed partial rank:", "between", round(low_rank), "and", round(high_rank))
        print("estimated rank quantiles:", low_q, "< ?? <", high_q)
        print("estimated true rank:", est_rank, "->", round(est_rank))
        print()

    # compute and return ndcg
    ndcg_score = np.log(2) / np.log(est_rank + 1)  # for single target, ndcg expression can be simplified
    ndcg_score = np.clip(ndcg_score, 0.0, 1.0)  # clip just in case

    return ndcg_score
