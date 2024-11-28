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
    cate_id: np.array
    brand_id: np.array
    customer_id: np.array
    campaign_id: np.array
    rel_ad_freqs: np.array


class UserBatch(NamedTuple):
    user: np.array


class TaobaoInteractionsSeqBatch(NamedTuple):
    user_feats: UserBatch
    ad_feats: np.array
    is_click: np.array
    timestamp: np.array
    is_padding: np.array

MAX_SEQ_LEN = 100

class TaobaoSequenceDataset(Dataset):

    def __init__(
        self, data_dir: str, 
        is_train: bool = True,
        min_timediff_unique: int = 30,       # The minimum number of seconds between identical interactions (user, adgroup, btag), or (user, cate, brand, btag), before they are considered duplicates
        min_training_interactions: int = 5,  # The minimum number of non-ad-click, browse, ad-click, favorite, add-to-cart, or purchase interactions required in a training sequence
        sequence_len: int = 100,
        slide_window_every: int = 100,
        user_features: list[str] = ["user", "gender", "age", "shopping", "occupation"],    # all features by default
        ad_features: list[str] = ["adgroup", "cate", "brand", "customer", "campaign"],     # all features by default
        force_reload: bool = False,
    ):
        user_profile_parquet = os.path.join(data_dir, f"user_profile.parquet")
        ad_feature_parquet = os.path.join(data_dir, f"ad_feature.parquet")
        assert os.path.isfile(user_profile_parquet), f"Cannot find user_profile file {user_profile_parquet}. Please generate using data_process notebooks"
        assert os.path.isfile(ad_feature_parquet), f"Cannot find ad_feature file {ad_feature_parquet}. Please generate using data_process notebooks"

        self.is_train = is_train
        self.user_feats = user_features
        self.ad_feats = ad_features
        self.selected_feats = [*user_features, *ad_features, "rel_ad_freq", "btag", "timestamp", "is_test", "seq_len"]

        sequence_params = f"timediff{min_timediff_unique}_mintrain{min_training_interactions}_seqlen{sequence_len}_slide{slide_window_every}"
        if force_reload:
            train_parquet = os.path.join(data_dir, "train.parquet")
            test_parquet = os.path.join(data_dir, "test.parquet")
            assert os.path.isfile(train_parquet), f"Cannot find train data file {train_parquet}. Please generate using data_process notebooks"
            assert os.path.isfile(test_parquet), f"Cannot find test data file {test_parquet}. Please generate using data_process notebooks"

            training_data = (pl.scan_parquet(train_parquet)
                .filter(pl.col("timediff").is_null() | (pl.col("timediff") >= min_timediff_unique))
                .filter(pl.len().over("user") >= min_training_interactions)
                .collect()
            )
            validation_data = (pl.scan_parquet(test_parquet)
                .filter(pl.col("user").is_in(training_data.select("user").unique()))
                .collect()
            )
            interactions: pl.DataFrame = pl.concat([training_data, validation_data], how="vertical", rechunk=True)
            del training_data, validation_data
            
            rel_ad_freqs = (interactions
                .filter(pl.col("adgroup") > -1)
                .select("adgroup", rel_ad_freq = (pl.len().over("adgroup") / pl.count("adgroup")).cast(pl.Float32))
                .unique()
            )
            sequences = (interactions
                .join(rel_ad_freqs, on="adgroup", how="left")
                .with_columns(pl.col("rel_ad_freq").fill_null(0.0))
                .group_by("user")
                .agg(
                    pl.col("gender", "age", "shopping", "occupation").first(),
                    pl.col("adgroup", "cate", "brand", "customer", "campaign", "rel_ad_freq", "btag", "timestamp", "is_test").sort_by("timestamp"),
                    seq_len = pl.col("btag").len().cast(pl.Int32)
                )
                .with_columns(pl.col("timestamp").list.diff().list.eval(pl.element().fill_null(0)))
            )
            del interactions, rel_ad_freqs

            max_seq_len = sequences.select(pl.col("seq_len").max()).item()
            train_sequences = (pl
                .concat([
                    (sequences
                        .filter(pl.col("seq_len") > abs(end_idx))
                        .select(
                            pl.col("user", "gender", "age", "shopping", "occupation"),
                            pl.col("adgroup", "cate", "brand", "customer", "campaign", "rel_ad_freq", "btag", "timestamp", "is_test")
                                .list.gather(range(end_idx-sequence_len, end_idx), null_on_oob=True)
                                .list.shift(pl.min_horizontal(pl.col("seq_len") + (end_idx-sequence_len), 0)),
                            seq_len = pl.min_horizontal(pl.col("seq_len") + end_idx, sequence_len).cast(pl.Int32)
                        )
                    ) for end_idx in range(-1, -max_seq_len, -slide_window_every)
                ], how="vertical")
                .filter(pl.col("seq_len") >= min_training_interactions)
                .with_columns(
                    pl.col("adgroup", "cate", "brand", "customer", "campaign").list.eval(pl.element().fill_null(-1)).list.to_array(sequence_len),
                    pl.col("rel_ad_freq").list.eval(pl.element().fill_null(0.0)).list.to_array(sequence_len),
                    pl.col("btag").list.eval(pl.element().fill_null(-2)).list.to_array(sequence_len),
                    pl.col("timestamp").list.eval(pl.element().fill_null(0)).list.to_array(sequence_len),
                    pl.col("is_test").list.eval(pl.element().fill_null(True)).list.to_array(sequence_len),
                )
            )
            train_sequences.write_parquet(os.path.join(data_dir, f"train_sequences_{sequence_params}.parquet"))
            if is_train:
                sequences = train_sequences.select(self.selected_feats).rechunk()
                del train_sequences
            
            test_sequences = (sequences
                .select(
                    pl.col("user", "gender", "age", "shopping", "occupation"),
                    pl.col("adgroup", "cate", "brand", "customer", "campaign", "rel_ad_freq", "btag", "timestamp", "is_test")
                        .list.gather(range(-sequence_len, 0), null_on_oob=True)
                        .list.shift(pl.min_horizontal(pl.col("seq_len") - sequence_len, 0)),
                    seq_len = pl.min_horizontal(pl.col("seq_len"), sequence_len).cast(pl.Int32)
                )
                .with_columns(
                    pl.col("adgroup", "cate", "brand", "customer", "campaign").list.eval(pl.element().fill_null(-1)).list.to_array(sequence_len),
                    pl.col("rel_ad_freq").list.eval(pl.element().fill_null(0.0)).list.to_array(sequence_len),
                    pl.col("btag").list.eval(pl.element().fill_null(-2)).list.to_array(sequence_len),
                    pl.col("timestamp").list.eval(pl.element().fill_null(0)).list.to_array(sequence_len),
                    pl.col("is_test").list.eval(pl.element().fill_null(True)).list.to_array(sequence_len),
                )
            )
            test_sequences.write_parquet(os.path.join(data_dir, f"test_sequences_{sequence_params}.parquet"))
            if not is_train:
                sequences = test_sequences.select(self.selected_feats).rechunk()
                del test_sequences

        else:
            train_seq_parquet = os.path.join(data_dir, f"train_sequences_{sequence_params}.parquet")
            test_seq_parquet = os.path.join(data_dir, f"test_sequences_{sequence_params}.parquet")
            assert os.path.isfile(train_seq_parquet), f"Cannot find train sequences file {train_seq_parquet}. Please generate by setting force_reload=True"
            assert os.path.isfile(test_seq_parquet), f"Cannot find test sequences file {test_seq_parquet}. Please generate by setting force_reload=True"
            if is_train:
                sequences = (pl
                    .scan_parquet(train_seq_parquet)
                    .select(self.selected_feats)
                    .collect()
                    .rechunk()
                )
            else:
                sequences = (pl
                    .scan_parquet(test_seq_parquet)
                    .select(self.selected_feats)
                    .collect()
                    .rechunk()
                )

        self.interaction_mapping = {-1: "ad_non_click" ,0: "browse", 1: "ad_click", 2: "favorite", 3: "add_to_cart", 4: "purchase"}

        self.user_profile = pl.scan_parquet(user_profile_parquet).select(self.user_feats).unique().collect()
        self.user_encoder = OrdinalEncoder(dtype=np.int32, encoded_missing_value=-1).fit(self.user_profile)
        self.user_encoder.set_output(transform="polars")

        self.ad_feature = pl.scan_parquet(ad_feature_parquet).select(self.ad_feats).unique().collect()
        self.ad_encoder = OrdinalEncoder(dtype=np.int32, encoded_missing_value=-1).fit(self.ad_feature)
        self.ad_encoder.set_output(transform="polars")

        self.user_data = sequences.select(self.user_feats).to_numpy()
        self.ads_data = [sequences[feat].to_numpy() for feat in self.ad_feats]
        self.rel_ad_freqs = sequences["rel_ad_freqs"].to_numpy()
        self.interaction_data = sequences["btag"].to_numpy()
        self.timestamps = sequences["timestamp"].to_numpy()
        self.padded_masks = sequences["is_test"].to_numpy()
        self.seq_lens = sequences["seq_len"].to_numpy()
        del sequences


    @cached_property
    def n_users(self):
        return len(self.user_profile["user"].unique())

    @cached_property
    def n_ads(self):
        # adgroups have nulls encoded as -1s
        return len(self.ad_feature["adgroup"].unique())-1
    
    @cached_property
    def n_brands(self):
        # brands have nulls encoded as -1s
        return len(self.ad_feature["brand"].unique())-1

    @cached_property
    def n_cates(self):
        return len(self.ad_feature["cate"].unique())
    
    def get_index(self):
        transformed_ad_feats: pl.DataFrame = self.ad_encoder.transform(self.ad_feature).sort("adgroup")
        batch = []
        for feat_name in self.ad_feats:
            batch.append(torch.tensor(transformed_ad_feats[feat_name].to_numpy()))
        batch.append(None)
        return AdBatch(*batch)
    
    def __len__(self):
        return len(self.seq_lens)
    
    def __getitem__(self, idx):
        max_batch_len = self.seq_lens[idx].max()
        return TaobaoInteractionsSeqBatch(
            UserBatch(self.user_data[idx]),
            AdBatch(*([ads_feat[idx, :max_batch_len] for ads_feat in self.ads_data]), self.rel_ad_freqs[idx]),
            self.interaction_data[idx, :max_batch_len],
            self.timestamps[idx, :max_batch_len],
            self.padded_masks[idx, :max_batch_len]
        )
