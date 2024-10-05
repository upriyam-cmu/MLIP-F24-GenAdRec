import torch
import pandas as pd
import numpy as np
from functools import cached_property
from torch.utils.data import Dataset
from typing import NamedTuple


class CategoricalFeature(NamedTuple):
    name: str
    num_classes: int


class UserBatch(NamedTuple):
    user: torch.Tensor


class AdBatch(NamedTuple):
    adgroup_id: torch.Tensor
    cate_id: torch.Tensor
    campaign_id: torch.Tensor
    brand: torch.Tensor


class InteractionsBatch(NamedTuple):
    user_feats: UserBatch
    ad_feats: AdBatch
    timestamp: torch.Tensor
    is_click: torch.Tensor


class RawInteractionsDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.raw_sample = pd.read_csv("../data/raw_sample.csv").drop(columns=["pid", "nonclk"])

        self.data = self._dedup_interactions()
        self.train_df, self.test_df = self._train_test_split(self.data)

    def _dedup_interactions(self):
        sorted_sample = self.raw_sample.sort_values(by=["user", "adgroup_id", "time_stamp"])
        timestamp_diff = sorted_sample["time_stamp"].diff().fillna(-1)
        user_diff = sorted_sample["user"].diff().fillna(-1)
        adgroup_diff = sorted_sample["adgroup_id"].diff().fillna(-1)
        deduped = sorted_sample[~((adgroup_diff == 0) & (user_diff == 0) & (timestamp_diff < 15 * 60))]
        return deduped
    
    def _train_test_split(self, df):
        clicks = df[df["clk"] == 1]
        clk_per_user = clicks.groupby(by="user")["clk"].count()
        max_clk_per_user = clk_per_user.max()

        click_cnt = clicks.sort_values(by=["user", "time_stamp"]).groupby("user")["clk"].rolling(max_clk_per_user, min_periods=1).sum()
        clicks = clicks.reset_index().drop(columns="index")
        clicks["clk_cnt"] = click_cnt.reset_index()["clk"]
        clicks = clicks.merge(clk_per_user.reset_index().rename({"clk":"clk_per_user"}, axis="columns"), on="user")
        split_timestamp = clicks[(clicks["clk_cnt"] == clicks["clk_per_user"]) & (clicks["clk_per_user"] > 1)][["user", "time_stamp"]]

        to_split = df.merge(split_timestamp.rename({"time_stamp": "split_timestamp"}, axis="columns"), on="user", how="left")
        test_filter = (to_split["clk"] == 1) & (to_split["split_timestamp"] <= to_split["time_stamp"])
        train_filter = (to_split["split_timestamp"] > to_split["time_stamp"]) | (to_split["split_timestamp"].isnull())
        to_split = to_split.drop(columns="split_timestamp")
        train_df, test_df = to_split[train_filter], to_split[test_filter]

        return train_df, test_df
    
    def get_train_test_split(self):
        return self.train_df, self.test_df

    def __getitem__(self, index):
        return self.data.iloc[index]


class InteractionsDataset(Dataset):
    def __init__(self, raw_interactions_dataset: RawInteractionsDataset, shuffle: bool = True, is_train: bool = True):
        idx = 0 if is_train else 1
        self.user_profile = pd.read_csv("../data/user_profile.csv").rename({"userid": "user"}, axis="columns")
        self.ad_feature = pd.read_csv("../data/ad_feature.csv")
        self.data = raw_interactions_dataset.get_train_test_split()[idx].merge(
            self.ad_feature, on="adgroup_id", how="left"
        ).merge(
            self.user_profile, on="user", how="left"
        )

        if shuffle:
            self.data = self.data.iloc[np.random.permutation(np.arange(len(self.data)))].reset_index().drop(columns="index")

    @cached_property
    def categorical_features(self):
        feat_names = AdBatch._fields
        max_values = self.data.max()
        return [CategoricalFeature(feat_name, max_values[feat_name]) for feat_name in feat_names]

    @cached_property
    def n_users(self):
        return self.data.max()["user"]
    
    def __getitem__(self, index) -> InteractionsBatch:
        data = self.data.iloc[index]

        user_feats = data[list(UserBatch._fields)]
        ad_feats = data[list(AdBatch._fields)]
        user_batch = UserBatch(*torch.tensor(user_feats.to_numpy()).split(1, dim=-1))
        ad_batch = AdBatch(*torch.tensor(ad_feats.to_numpy()).split(1, dim=-1))
        timestamp = torch.tensor(data["time_stamp"].to_numpy()).unsqueeze(1)
        clk = torch.tensor(data["clk"].to_numpy()).unsqueeze(1)
        
        return InteractionsBatch(
            user_feats=user_batch,
            ad_feats=ad_batch,
            timestamp=timestamp,
            is_click=clk
        )