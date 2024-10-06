import os
import numpy as np
import polars as pl
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data.dataset import Dataset

class TaobaoUserClicksDataset(Dataset):
    def __init__(
        self, data_dir, training, filter_clicks, num_validation=1,
        include_user_ids = True,
        user_features = [], #, "final_gender_code", "age_level", "pvalue_level", "shopping_level", "occupation", "new_user_class_level"],
        include_ad_ids = False,
        ad_features = ["cate_id", "brand", "customer", "campaign_id"],         
    ):
        assert include_user_ids or user_features, "must specify at least one user feature to include"
        assert include_ad_ids or ad_features, "must specify at least one ad feature to include"
        
        raw_samples = pl.scan_csv(os.path.join(data_dir, "raw_sample.csv"))
        if filter_clicks:
            raw_samples = raw_samples.filter(pl.col("clk") == 1)
        
        self.user_feats = (["user"] if include_user_ids else []) + user_features
        self.ad_feats = (["adgroup_id"] if include_ad_ids else []) + ad_features
        
        user_clicks = (
            raw_samples
            .drop("pid")
            .unique()
            .select("user", "time_stamp", "adgroup_id", "clk")
            .join(
                other=pl.scan_csv(os.path.join(data_dir, "user_profile.csv")).select(["userid"] + user_features),
                left_on="user",
                right_on="userid",
            )
            .join(
                other=pl.scan_csv(os.path.join(data_dir, "ad_feature.csv")).select(["adgroup_id"] + ad_features),
                on="adgroup_id",
            )
            .sort("user", "time_stamp", nulls_last=True)
            .group_by("user", maintain_order=True)
            .agg(
                (pl.max_horizontal(pl.len().cast(int) - num_validation, 0) if training else
                    pl.min_horizontal(pl.len(), num_validation)).alias("included_clicks"),
                (pl.all().head(pl.max_horizontal(pl.len().cast(int) - num_validation, 0)) if training else
                    pl.all().tail(pl.min_horizontal(pl.len(), num_validation))),
            )
            .filter(pl.col("included_clicks") > 0)
            .explode(pl.all().exclude("user", "included_clicks"))
            .select(self.user_feats + self.ad_feats + ["time_stamp", "clk"])
            .collect()
        ).to_numpy()
        
        user_profile = (
            pl.scan_csv(os.path.join(data_dir, "user_profile.csv"))
            .select((["userid"] if include_user_ids else []) + user_features)
            .unique()
            .collect()
        ).to_numpy()
        self.user_encoder = OrdinalEncoder().fit(user_profile)
        self.user_data = self.user_encoder.transform(user_clicks[:, :len(self.user_feats)])
        
        ad_feature = (
            pl.scan_csv(os.path.join(data_dir, "ad_feature.csv"))
            .select((["adgroup_id"] if include_ad_ids else []) + ad_features)
            .unique()
            .collect()
        ).to_numpy()
        self.ads_encoder = OrdinalEncoder().fit(ad_feature)
        self.ads_data = self.ads_encoder.transform(user_clicks[:, len(self.user_feats):-2])
        
        self.timestamps = user_clicks[:, -2].astype(int)
        self.clicks = user_clicks[:, -1].astype(int)
        
    def __len__(self):
        return len(self.clicks)

    def __getitem__(self, idx):
        return self.user_data[idx], self.ads_data[idx], self.timestamps[idx], self.clicks[idx]
 