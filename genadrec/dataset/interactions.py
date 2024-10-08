import torch
import pandas as pd
import numpy as np
from functools import cached_property
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import Dataset
from typing import NamedTuple
import time
import os


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
    q_proba: torch.Tensor


class InteractionsBatch(NamedTuple):
    user_feats: UserBatch
    ad_feats: AdBatch
    train_index: AdBatch
    timestamp: torch.Tensor
    is_click: torch.Tensor


class InteractionsDataset(Dataset):
    TRAIN_FILENAME = "train_split.csv"
    EVAL_FILENAME = "eval_split.csv"
    
    def __init__(self, path: str, shuffle: bool = True, is_train: bool = True, train_index_size: int = 2048):
        super().__init__()

        self.raw_sample = pd.read_csv("data/raw_sample.csv").drop(columns=["pid", "nonclk"])

        idx = 0 if is_train else 1
        self.is_train = is_train
        self.train_index_size = train_index_size
        self.user_profile = pd.read_csv("data/user_profile.csv").rename({"userid": "user"}, axis="columns")
        self.ad_feature = pd.read_csv("data/ad_feature.csv")

        train_file_exists = os.path.isfile(path + self.TRAIN_FILENAME)
        eval_file_exists = os.path.isfile(path + self.EVAL_FILENAME)
    
        if not train_file_exists or not eval_file_exists:
            self.data = self._dedup_interactions()
            train_test_split = self._train_test_split(self.data)
            self.data = train_test_split[idx].merge(
                self.ad_feature, on="adgroup_id", how="left"
            ).merge(
                self.user_profile, on="user", how="left"
            )

            self.train_data = train_test_split[0].merge(
                self.ad_feature, on="adgroup_id", how="left"
            ).merge(
                self.user_profile, on="user", how="left"
            )

            self.data = self.with_ads_occ_proba(self.data)

            self.data = self.encode_categories(self.train_data, self.data)

            self.data.to_csv(path + self.TRAIN_FILENAME if is_train else path + self.EVAL_FILENAME)
            self.ad_feature.to_csv("data/ad_feature.csv")
        
        else:
            self.train_data = pd.read_csv(path + self.TRAIN_FILENAME)
            data_file_path = path + self.TRAIN_FILENAME if self.is_train else path + self.EVAL_FILENAME
            self.data = pd.read_csv(data_file_path)

        if shuffle:
            self.data = self.data.iloc[np.random.permutation(np.arange(len(self.data)))].reset_index().drop(columns="index")
    
    def _dedup_interactions(self):
        sorted_sample = self.raw_sample.sort_values(by=["user", "adgroup_id", "time_stamp"])
        timestamp_diff = sorted_sample["time_stamp"].diff().fillna(-1)
        user_diff = sorted_sample["user"].diff().fillna(-1)
        adgroup_diff = sorted_sample["adgroup_id"].diff().fillna(-1)
        deduped = sorted_sample[~((adgroup_diff == 0) & (user_diff == 0) & (timestamp_diff < 15 * 60))]

        deduped = deduped.merge(
            deduped[deduped["clk"] == 1].groupby("user")["adgroup_id"].count().reset_index().rename({"adgroup_id": "clk_pu"}, axis="columns"),
            on="user", how="left"
        )
        deduped = deduped[deduped["clk_pu"] > 2]
        return deduped
    
    def _train_test_split(self, df):
        clicks = df[df["clk"] == 1]
        clk_per_user = clicks.groupby(by="user")["clk"].count()
        max_clk_per_user = clk_per_user.max()

        click_cnt = (
            clicks.sort_values(["time_stamp"], ascending=True)
              .groupby("user")
              .cumcount() + 1
        )

        clicks["clk_cnt"] = click_cnt
        clicks = clicks.reset_index().drop(columns="index")
        clicks = clicks.merge(clk_per_user.reset_index().rename({"clk":"clk_per_user"}, axis="columns"), on="user")
        split_timestamp = clicks[(clicks["clk_cnt"] == clicks["clk_per_user"]) & (clicks["clk_per_user"] > 1)][["user", "time_stamp"]]

        to_split = df.merge(split_timestamp.rename({"time_stamp": "split_timestamp"}, axis="columns"), on="user", how="left")
        test_filter = (to_split["clk"] == 1) & (to_split["split_timestamp"] <= to_split["time_stamp"])
        train_filter = (to_split["split_timestamp"] > to_split["time_stamp"]) | (to_split["split_timestamp"].isnull())
        to_split = to_split.drop(columns="split_timestamp")
        train_df, test_df = to_split[train_filter], to_split[test_filter]

        return train_df, test_df

    def encode_categories(self, train, target):
        user_encoder = OrdinalEncoder(dtype=np.int32)
        ad_feats = [feat for feat in AdBatch._fields if feat != "q_proba"]
        for feat in ad_feats:
            ad_encoder = OrdinalEncoder(dtype=np.int32, handle_unknown='use_encoded_value', unknown_value=len(train[feat].unique())+1)
            ad_encoder.fit(train[feat].fillna(-1).to_numpy().reshape(-1, 1))
            target[feat] = ad_encoder.transform(target[feat].fillna(-1).to_numpy().reshape(-1, 1))
            self.ad_feature[feat] = ad_encoder.transform(self.ad_feature[feat].fillna(-1).to_numpy().reshape(-1, 1))
        user_encoder.fit(train[list(UserBatch._fields)])
        target = target[target["user"].isin(train["user"].unique())]
        target[list(UserBatch._fields)] = user_encoder.transform(target[list(UserBatch._fields)])
        return target

    @cached_property
    def categorical_features(self):
        feat_names = AdBatch._fields
        return [
            CategoricalFeature(feat_name, len(self.train_data[feat_name].unique())+2) for feat_name in feat_names 
            if feat_name not in ("adgroup_id", "q_proba")
        ]

    @cached_property
    def n_users(self):
        return int(self.data.max()["user"]+1)
    
    def with_ads_occ_proba(self, data):
        cnt = data.groupby("adgroup_id")["clk"].count().reset_index().rename({"clk": "q_proba"}, axis="columns")
        cnt["q_proba"] = cnt["q_proba"] / len(self)
        return data.merge(
            cnt,
            on="adgroup_id"
        )
    
    def get_index(self, train_index_size: int = 1):
        if not self.is_train:
            self.ad_feature["q_proba"] = 0
            sample_feats = self.ad_feature[list(AdBatch._fields)]
        else:
            sample_feats = self.data[list(AdBatch._fields)].sample(train_index_size, replace=True)
        index = AdBatch(*torch.tensor(sample_feats.fillna(0).to_numpy().astype(np.int32).T).split(1, dim=0))
        return index
    
    def __getitem__(self, index) -> InteractionsBatch:
        start = time.time()
        data = self.data.iloc[index]

        user_feats = data[list(UserBatch._fields)]
        ad_feats = data[list(AdBatch._fields)]
        user_batch = UserBatch(*torch.tensor(user_feats.to_numpy().astype(np.int32).T).split(1, dim=0))
        ad_batch = AdBatch(*torch.tensor(ad_feats.fillna(0).to_numpy().T).split(1, dim=0))
        timestamp = [data["time_stamp"]] if not isinstance(data["time_stamp"], pd.Series) else data["time_stamp"].to_numpy()
        timestamp = torch.tensor(timestamp).to(torch.int32)
        clk = [data["clk"]] if not isinstance(data["clk"], pd.Series) else data["clk"].to_numpy()
        clk = torch.tensor(clk).to(torch.int32)
        
        train_index = self.get_index(self.train_index_size)
        end = time.time()
        #import pdb; pdb.set_trace()
        #print(f"Getitem time: {end - start}")
        return InteractionsBatch(
            user_feats=user_batch,
            ad_feats=ad_batch,
            train_index=train_index,
            timestamp=timestamp,
            is_click=clk,
        )
    
    def __len__(self):
        return len(self.data)