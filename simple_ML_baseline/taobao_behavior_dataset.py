import os
import numpy as np
import polars as pl
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
        
        self.conditional_masking = conditional_masking
        self.sequence_mode = sequence_mode
        
        train_parquet = os.path.join(data_dir, f"train_min_{min_ad_clicks}_click.parquet")
        test_parquet = os.path.join(data_dir, f"test_min_{min_ad_clicks}_click.parquet")
        
        assert os.path.isfile(train_parquet), f"Cannot find train data file {train_parquet}. Please generate using data_preprocess.ipynb"
        assert os.path.isfile(test_parquet), f"Cannot find test data file {test_parquet}. Please generate using data_preprocess.ipynb"
        
        train_data = pl.read_parquet(train_parquet)
        test_data = pl.read_parquet(test_parquet)

        self.user_feats = list(user_features)
        self.user_profile = pl.concat([
            train_data.select(self.user_feats).unique(),
            test_data.select(self.user_feats).unique(),
        ]).unique()
        self.user_encoder = OrdinalEncoder(dtype=np.uint32).fit(self.user_profile)
        self.user_encoder.set_output(transform="polars")

        self.ad_feats = list(ad_features)
        self.ad_feature = pl.concat([
            train_data.select(self.ad_feats).unique(),
            test_data.select(self.ad_feats).unique(),
        ]).unique()
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
        
        if mode == "pretrain":
            self.raw_data = train_data.filter(pl.col("adgroup").is_null())
        elif mode == "finetune":
            self.raw_data = train_data.drop_nulls("adgroup")
        elif mode == "train":
            self.raw_data = train_data
        elif mode == "test":
            self.raw_data = test_data
            
        self.user_data = self.user_encoder.transform(self.raw_data.select(self.user_feats))
        self.ads_data = self.ad_encoder.transform(self.raw_data.select(self.ad_feats))
        
        self.interaction_mapping = {-1: "non_ad_click", 0: "browse", 1: "ad_click", 2: "favorite", 3: "add_to_cart", 4: "purchase"}
        self.interaction_data = self.raw_data.select("btag", "timestamp")
        
        self.transformed_data = (pl
            .concat([self.user_data, self.ads_data, self.interaction_data], how="horizontal")
            .select(pl.all(), (pl.len().over("adgroup") / len(self.interaction_data)).cast(pl.Float32).alias("rel_ad_freq"))
        )
        
        if sequence_mode:
            user_features.remove("user")
            sequences = (self.transformed_data
                .sort("user", "timestamp")
                .group_by("user", maintain_order=True)
                .agg(
                    pl.col(user_features).first(), 
                    pl.col(*self.ad_feats, "rel_ad_freq", "btag", "timestamp"), 
                    seq_len=pl.col("btag").len()
                )
            )
            max_seq_len = sequences.select(pl.col("seq_len").max()).item()
            self.sequence_data = (sequences
                .with_columns(pad_len=max_seq_len-pl.col("seq_len"))
                .select(
                    pl.col(self.user_feats),
                    *(pl.col(feat).list.concat(
                        pl.lit(0, dtype=pl.UInt32).repeat_by(pl.col("pad_len"))
                    ).list.to_array(max_seq_len) for feat in [*self.ad_feats, "rel_ad_freq", "btag", "timestamp"]),
                    padded_mask = pl.lit(False).repeat_by(pl.col("seq_len")).list.concat(
                        pl.lit(True).repeat_by(pl.col("pad_len"))
                    ).list.to_array(max_seq_len)
                )
            )
            self.user_data = self.sequence_data.select(self.user_feats).to_numpy().squeeze()
            self.ads_data = [self.sequence_data.select(feat).to_series().to_numpy() for feat in (*self.ad_feats, "rel_ad_freq")]
            self.interaction_data = self.sequence_data.select("btag").to_series().to_numpy()
            self.timestamps = self.sequence_data.select("timestamp").to_series().to_numpy()
            self.padded_masks = self.sequence_data.select("padded_mask").to_series().to_numpy()
        else:
            self.user_data = self.user_data.to_numpy().squeeze()
            self.ads_data = self.ads_data.to_numpy().squeeze()
            self.interaction_data = self.interaction_data.to_series().to_numpy()
            self.timestamps = self.timestamps.to_series().to_numpy()
        
        del train_data
        del test_data
    
    @cached_property
    def n_users(self):
        return self.transformed_data["user"].max()+1

    @cached_property
    def n_ads(self):
        return self.transformed_data["adgroup"].max()+1
    
    def __len__(self):
        return len(self.timestamps)
    
    def __getitem__(self, idx):
        if self.sequence_mode:
            max_batch_len = (~self.padded_masks[idx]).sum(axis=1).max()
            return TaobaoInteractionsSeqBatch(
                UserBatch(self.user_data[idx].astype(np.int32)),
                AdBatch(*([ads_feat[idx, :max_batch_len] for ads_feat in self.ads_data])), # if len(self.ad_feats) > 1 else [self.ads_data[idx, :max_batch_len].astype(np.int32)])),
                self.interaction_data[idx, :max_batch_len],
                self.timestamps[idx, :max_batch_len].astype(np.int32),
                self.padded_masks[idx, :max_batch_len]
            )
        else:
            user_data, ads_data, timestamps, interactions = self.user_data[idx], self.ads_data[idx], self.timestamps[idx], self.interaction_data[idx]
            ads_masks = []
            ad_feats_start = 1 if self.include_ad_ids else 0
            for i, dim in enumerate(self.output_dims[1+ad_feats_start:]):
                ad_feats_end = i + 1 + ad_feats_start
                mask_indices = self.conditional_mappings[i][tuple(ads_data[ad_feats_start:ad_feats_end].tolist())]
                mask = np.ones(dim, dtype=bool)
                mask[mask_indices] = False
                ads_masks.append(mask)
            return user_data, ads_data, ads_masks, timestamps, interactions