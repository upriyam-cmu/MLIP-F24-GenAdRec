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
        assert num_validation >= 0, "num_validation must be non-negative"
        assert include_user_ids or user_features, "must specify at least one user feature to include"
        assert include_ad_ids or ad_features, "must specify at least one ad feature to include"
        
        self.include_user_ids = include_user_ids
        self.include_ad_ids = include_ad_ids
        
        raw_samples = pl.scan_csv(os.path.join(data_dir, "raw_sample.csv"))
        if filter_clicks:
            raw_samples = raw_samples.filter(pl.col("clk") == 1)
        
        self.user_feats = (["user"] if include_user_ids else []) + user_features
        self.ad_feats = (["adgroup_id"] if include_ad_ids else []) + ad_features
        
        click_data = (
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
            .filter(
                pl.len().over("user") > num_validation
            )
            .collect()
        )
        self.user_encoder = OrdinalEncoder(dtype=np.int64).fit(click_data.select(self.user_feats))
        self.ad_encoder = OrdinalEncoder(dtype=np.int64).fit(click_data.select(self.ad_feats))

        self.input_dims = [user.shape[0] for user in self.user_encoder.categories_]
        self.output_dims = [category.shape[0] for category in self.ad_encoder.categories_]

        self.ad_encoder.set_output(transform="polars")
        ad_feats: pl.DataFrame = self.ad_encoder.transform(click_data.select(self.ad_feats))
        self.ad_features = ad_feats.unique().to_numpy()
        self.ad_encoder.set_output(transform="default")

        self.conditional_mappings = []
        for i in range(1, len(ad_features)):
            conditional_map = (
                ad_feats
                .select(ad_features[:i+1])
                .group_by(ad_features[:i])
                .agg(
                    pl.col(ad_features[i]).unique()
                )
                .to_pandas()
            )
            conditional_map.index = list(zip(*[conditional_map[ad_features[j]] for j in range(i)]))
            self.conditional_mappings.append(conditional_map.to_dict()[ad_features[i]])

        user_clicks = (
            click_data
            .sort("user", "time_stamp", nulls_last=True)
            .group_by("user", maintain_order=True)
            .agg(
                pl.all().head(pl.len() - num_validation) if training else pl.all().tail(num_validation),
            )
            .explode(pl.all().exclude("user"))
            .select(self.user_feats + self.ad_feats + ["time_stamp", "clk"])
            .to_numpy()
        )
        
        self.user_data = self.user_encoder.transform(user_clicks[:, :len(self.user_feats)])
        self.ads_data = self.ad_encoder.transform(user_clicks[:, len(self.user_feats):-2])
        
        self.timestamps = user_clicks[:, -2].astype(int)
        self.clicks = user_clicks[:, -1].astype(bool)

    def __len__(self):
        return len(self.clicks)

    def __getitem__(self, idx):
        user_data, ads_data, timestamps, clicks = self.user_data[idx], self.ads_data[idx], self.timestamps[idx], self.clicks[idx]
        ads_masks = []
        ad_feats_start = 1 if self.include_ad_ids else 0
        for i, dim in enumerate(self.output_dims[1+ad_feats_start:]):
            ad_feats_end = i + 1 + ad_feats_start
            mask_indices = self.conditional_mappings[i][tuple(ads_data[ad_feats_start:ad_feats_end].tolist())]
            mask = np.ones(dim, dtype=bool)
            mask[mask_indices] = False
            ads_masks.append(mask)
        return user_data, ads_data, ads_masks, timestamps, clicks
