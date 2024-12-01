import os
import numpy as np
import polars as pl
from functools import cached_property
from encoder.polars_ordinal_encoder import PolarsOrdinalEncoder
from torch.utils.data.dataset import Dataset
from typing import NamedTuple


class UserBatch(NamedTuple):
    user: np.array
    gender: np.array
    age: np.array
    shopping: np.array
    occupation: np.array


class UserFeaturesDataset(Dataset):

    def __init__(
        self, data_dir: str,
        user_features: list[str] = ["user", "gender", "age", "shopping", "occupation"],
    ):
        self.user_feats = user_features

        user_profile_parquet = os.path.join(data_dir, f"user_profile.parquet")
        assert os.path.isfile(user_profile_parquet), f"Cannot find user_profile file {user_profile_parquet}. Please generate using data_process notebooks"
        self.user_profile = pl.read_parquet(user_profile_parquet)
        self.user_encoder = PolarsOrdinalEncoder(fit_data = self.user_profile)

        self.transformed_user_data = self.user_encoder.transform(self.user_profile.drop_nulls())
        self.user_data = {feat: self.transformed_user_data[feat].to_numpy() for feat in self.user_feats}


    @cached_property
    def n_users(self):
        return self.user_encoder.feat_num_unique_with_null["user"]

    def __len__(self):
        return len(self.transformed_user_data)

    def __getitem__(self, idx):
        return UserBatch(**{feat: self.user_data[feat][idx] for feat in self.user_feats})
