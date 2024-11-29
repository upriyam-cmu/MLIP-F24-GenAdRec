import os
import numpy as np
import polars as pl
import torch
from functools import cached_property
from encoder.polars_ordinal_encoder import PolarsOrdinalEncoder
from torch.utils.data.dataset import Dataset
from typing import NamedTuple


class AdBatch(NamedTuple):
    adgroup_id: np.array
    cate_id: np.array
    brand_id: np.array
    customer_id: np.array
    campaign_id: np.array
    rel_ad_freqs: np.array


class UserBatch(NamedTuple):
    user: np.array


class TaobaoInteractionsSeqBatch(NamedTuple):
    user_feats: UserBatch
    ad_feats: AdBatch
    is_click: np.array
    timestamp: np.array
    is_padding: np.array


class TaobaoSequenceDataset(Dataset):

    def __init__(
        self, data_dir: str,
        is_train: bool = True,
        min_timediff_unique: int = 30,      # The minimum number of seconds between identical interactions (user, adgroup, btag), or (user, cate, brand, btag), before they are considered duplicates
        min_training_interactions: int = 5, # The minimum number of non-ad-click, browse, ad-click, favorite, add-to-cart, or purchase interactions required in a training sequence
        augmented: bool = False,            # Whether to include behavior log interaction data or not
        sequence_len: int = 128,
        slide_window_every: int = 64,
        user_features: list[str] = ["user", "gender", "age", "shopping", "occupation"],    # all features by default
        ad_features: list[str] = ["adgroup", "cate", "brand", "customer", "campaign"],     # all features by default
    ):
        user_profile_parquet = os.path.join(data_dir, f"user_profile.parquet")
        ad_feature_parquet = os.path.join(data_dir, f"ad_feature.parquet")
        assert os.path.isfile(user_profile_parquet), f"Cannot find user_profile file {user_profile_parquet}. Please generate using data_process notebooks"
        assert os.path.isfile(ad_feature_parquet), f"Cannot find ad_feature file {ad_feature_parquet}. Please generate using data_process notebooks"

        self.is_train = is_train
        self.user_feats = user_features
        self.ad_feats = ad_features
        self.missing_ad_feats = set(["adgroup", "cate", "brand", "customer", "campaign"]).difference(self.ad_feats)
        
        self.interaction_mapping = {-1: "ad_non_click" ,0: "browse", 1: "ad_click", 2: "favorite", 3: "add_to_cart", 4: "purchase"}

        self.user_profile = pl.read_parquet(user_profile_parquet)
        self.user_encoder = PolarsOrdinalEncoder(fit_data = self.user_profile)

        self.ad_feature = pl.read_parquet(ad_feature_parquet)
        self.ad_encoder = PolarsOrdinalEncoder(fit_data = self.ad_feature)

        train_sequence_params = f"timediff{min_timediff_unique}_mintrain{min_training_interactions}_seqlen{sequence_len}_slide{slide_window_every}" + ("_aug" if augmented else "")
        test_sequence_params = f"timediff{min_timediff_unique}_mintrain{min_training_interactions}_seqlen{sequence_len}" + ("_aug" if augmented else "")

        train_file = os.path.join(data_dir, f"train_data_{train_sequence_params}.npz")
        test_file = os.path.join(data_dir, f"test_data_{test_sequence_params}.npz")
        assert os.path.isfile(train_file), f"Cannot find training data {train_file}. Please generate with data_process notebooks"
        assert os.path.isfile(test_file), f"Cannot find test data {test_file}. Please generate with data_process notebooks"
        
        if self.is_train:
            data = np.load(train_file)
        else:
            data = np.load(test_file)

        user_feat_map = {"user": 0, "gender": 1, "age": 2, "shopping": 3, "occupation": 4}
        user_feat_indices = [user_feat_map[feat] for feat in self.user_feats]
        self.user_data = data["user_data"][:, user_feat_indices]
        self.ads_data = {feat: data[feat] for feat in self.ad_feats}
        self.rel_ad_freqs = data["rel_ad_freqs"]
        self.interaction_data = data["interaction_data"]
        self.timestamps = data["timestamps"]
        self.padded_masks = data["padded_masks"]
        self.seq_lens = data["seq_lens"]
        del data


    @cached_property
    def n_users(self):
        return len(self.user_profile["user"].unique())

    @cached_property
    def n_ads(self):
        return len(self.ad_feature["adgroup"].unique())

    @cached_property
    def n_brands(self):
        return len(self.ad_feature["brand"].unique())

    @cached_property
    def n_cates(self):
        return len(self.ad_feature["cate"].unique())

    def get_index(self):
        transformed_ad_feats = self.ad_encoder.transform(self.ad_feature.filter(pl.col("adgroup") > -1)).sort("adgroup")
        return AdBatch(**{feat+"_id": torch.tensor(transformed_ad_feats[feat].to_numpy()) for feat in self.ad_feats},
                       **{feat+"_id": None for feat in self.missing_ad_feats},
                       rel_ad_freqs=None)

    def __len__(self):
        return len(self.seq_lens)

    def __getitem__(self, idx):
        max_batch_len = self.seq_lens[idx].max()
        return TaobaoInteractionsSeqBatch(
            UserBatch(self.user_data[idx]),
            AdBatch(**{feat+"_id": self.ads_data[feat][idx, :max_batch_len] for feat in self.ad_feats},
                    **{feat+"_id": None for feat in self.missing_ad_feats},
                    rel_ad_freqs=self.rel_ad_freqs[idx, :max_batch_len]),
            self.interaction_data[idx, :max_batch_len],
            self.timestamps[idx, :max_batch_len],
            self.padded_masks[idx, :max_batch_len]
        )
