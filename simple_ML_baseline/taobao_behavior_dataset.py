import os
import numpy as np
import polars as pl
import torch
from functools import cached_property
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data.dataset import Dataset
from typing import NamedTuple


class AdBatch(NamedTuple):
    adgroup_id: np.array
    rel_ad_freqs: np.array


class UserBatch(NamedTuple):
    user: np.array


class TaobaoInteractionsSeqBatch(NamedTuple):
    user_feats: UserBatch
    ad_feats: np.array
    is_click: np.array
    timestamp: np.array
    is_padding: np.array

MAX_SEQ_LEN = 200

class TaobaoDataset(Dataset):

    def __init__(
        self, data_dir, min_ad_clicks, mode = "train", sequence_mode = False,
        user_features = ["user", "gender", "age", "shopping", "occupation"],    # all features by default
        ad_features = ["cate", "brand", "customer", "campaign", "adgroup"],     # all features by default
        conditional_masking = False    # maps ad feature tuples to next feature subset in same order as provided
    ):
        assert mode in ["pretrain", "finetune", "train", "test"], "mode must be pretrain, finetune, train, or test"
        assert not (conditional_masking and sequence_mode), "Can only support one of conditional masking and sequence mode at a time"
        assert "user" in user_features, f"Missing user id in user features: {user_features}"
        assert "adgroup" in ad_features, f"Missing ad id in ad features: {ad_features}"
        
        user_profile_parquet = os.path.join(data_dir, f"user_profile_{min_ad_clicks}.parquet")
        ad_feature_parquet = os.path.join(data_dir, f"ad_feature_{min_ad_clicks}.parquet")
        interactions_parquet = os.path.join(data_dir, f"interaction_seq_{min_ad_clicks}.parquet" if sequence_mode else f"interactions_{min_ad_clicks}.parquet")
        
        assert os.path.isfile(user_profile_parquet), f"Cannot find user_profile file {user_profile_parquet}. Please generate using data_preprocess_encode.ipynb"
        assert os.path.isfile(ad_feature_parquet), f"Cannot find ad_feature file {ad_feature_parquet}. Please generate using data_preprocess_encode.ipynb"
        assert os.path.isfile(interactions_parquet), f"Cannot find interactions file {interactions_parquet}. Please generate using data_preprocess.ipynb"

        self.mode = mode
        self.interaction_mapping = {-1: "ad_non_click" ,0: "browse", 1: "ad_click", 2: "favorite", 3: "add_to_cart", 4: "purchase"}
        self.conditional_masking = conditional_masking
        self.sequence_mode = sequence_mode

        self.user_feats = list(user_features)
        self.user_profile = pl.read_parquet(user_profile_parquet).select(self.user_feats).unique()
        self.user_encoder = OrdinalEncoder(dtype=np.uint32).fit(self.user_profile)
        self.user_encoder.set_output(transform="polars")

        self.ad_feats = list(ad_features)
        self.ad_feature = pl.read_parquet(ad_feature_parquet).select(self.ad_feats).unique()
        self.ad_encoder = OrdinalEncoder(dtype=np.int32, encoded_missing_value=-1).fit(self.ad_feature)
        self.ad_encoder.set_output(transform="polars")

        self.input_dims = [user.shape[0] for user in self.user_encoder.categories_]
        self.output_dims = [category.shape[0] for category in self.ad_encoder.categories_]
        
        if self.conditional_masking:
            polars_transformed_ad_feats: pl.DataFrame = self.ad_encoder.transform(self.ad_feature)
            self.ad_features = polars_transformed_ad_feats.unique().to_numpy()
        
            self.conditional_mappings = []
            for i in range(1, len(self.ad_feats)):
                conditional_map = (
                    polars_transformed_ad_feats
                    .select(self.ad_feats[:i+1])
                    .group_by(self.ad_feats[:i])
                    .agg(
                        pl.col(self.ad_feats[i]).unique()
                    )
                    .to_pandas()
                )
                conditional_map.index = list(zip(*[conditional_map[self.ad_feats[j]] for j in range(i)]))
                self.conditional_mappings.append(conditional_map.to_dict()[self.ad_feats[i]])
        
        interactions = pl.read_parquet(interactions_parquet)
        if sequence_mode:
            self.user_data = interactions.select(self.user_feats).to_numpy().squeeze()
            self.seq_lens = interactions.select("seq_len").to_series().to_numpy()
            self.ads_data = [interactions.select(feat).to_series().to_numpy() for feat in (*self.ad_feats, "rel_ad_freq")]
            self.interaction_data = interactions.select("btag").to_series().to_numpy()
            self.timestamps = interactions.select("timestamp").to_series().to_numpy()
            self.padded_masks = interactions.select("padded_mask").to_series().to_numpy()
            self.test_indices = interactions.select("is_test").to_series().to_numpy()
            if self.mode != "test":
                test_indices = np.nonzero(self.test_indices)
                for ads_feat in self.ads_data:
                    ads_feat[test_indices] = 0
                self.interaction_data[test_indices] = 0
                self.timestamps[test_indices] = 0
                self.padded_masks[test_indices] = True
                self.seq_lens = self.seq_lens - 1
        else:
            if mode == "pretrain":
                interactions = interactions.filter(pl.col("adgroup").is_null())
            elif mode == "finetune":
                interactions = interactions.drop_nulls("adgroup").filter(~pl.col("is_test"))
            elif mode == "train":
                interactions = interactions.filter(~pl.col("is_test"))
            elif mode == "test":
                interactions = interactions.filter(pl.col("is_test"))
            
            self.user_data = interactions.select(self.user_feats).to_numpy().squeeze()
            self.ads_data = interactions.select(self.ad_feats).to_numpy().squeeze()
            self.interaction_data = interactions.select("btag").to_series().to_numpy()
            self.timestamps = interactions.select("timestamp").to_series().to_numpy()
        
        del interactions
    
    @cached_property
    def n_users(self):
        return len(self.user_profile["user"].unique())+1

    @cached_property
    def n_ads(self):
        return len(self.ad_feature["adgroup"].unique())+1
    
    def get_index(self):
        ad_index = torch.arange(self.n_ads)
        return AdBatch(adgroup_id=ad_index, rel_ad_freqs=None)
    
    def __len__(self):
        return len(self.timestamps)
    
    def __getitem__(self, idx):
        if self.sequence_mode:
            max_batch_len = self.seq_lens[idx].max()
            return TaobaoInteractionsSeqBatch(
                UserBatch(self.user_data[idx].astype(np.int32)),
                AdBatch(*([ads_feat[idx, :max_batch_len] for ads_feat in self.ads_data])),
                self.interaction_data[idx, :max_batch_len],
                self.timestamps[idx, :max_batch_len].astype(np.int32),
                self.padded_masks[idx, :max_batch_len]
            )
        else:
            user_data, ads_data, timestamps, interactions = self.user_data[idx], self.ads_data[idx], self.timestamps[idx], self.interaction_data[idx]
            ads_masks = []
            for i, dim in enumerate(self.output_dims[1:]):
                mask_indices = self.conditional_mappings[i][tuple(ads_data[:i+1].tolist())]
                mask = np.ones(dim, dtype=bool)
                mask[mask_indices] = False
                ads_masks.append(mask)
            return user_data, ads_data, ads_masks, timestamps, interactions