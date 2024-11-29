import os
import numpy as np
import polars as pl
from encoder.polars_ordinal_encoder import PolarsOrdinalEncoder
from torch.utils.data.dataset import Dataset

class TaobaoDataset(Dataset):

    def __init__(
        self, data_dir,
        mode = "train",
        min_timediff_unique: int = 30,      # The minimum number of seconds between identical interactions (user, adgroup, btag), or (user, cate, brand, btag), before they are considered duplicates
        min_training_interactions: int = 5, # The minimum number of non-ad-click, browse, ad-click, favorite, add-to-cart, or purchase interactions required in a training sequence
        augmented: bool = False,            # Whether to include behavior log interaction data or not
        user_features = ["user", "gender", "age", "shopping", "occupation"],    # all features by default
        ad_features = ["cate", "brand", "customer", "campaign", "adgroup"],     # all features by default
    ):
        assert mode in ["pretrain", "finetune", "train", "test"], "mode must be pretrain, finetune, train, or test"
        assert mode != "pretrain" or augmented, "if pretraining, loaded dataset must be augmented"

        dataset_params = f"timediff{min_timediff_unique}_mintrain{min_training_interactions}" + ("_aug" if augmented else "")

        user_profile_parquet = os.path.join(data_dir, f"user_profile.parquet")
        ad_feature_parquet = os.path.join(data_dir, f"ad_feature.parquet")
        train_file = os.path.join(data_dir, f"train_data_{dataset_params}.npz")
        test_file = os.path.join(data_dir, f"test_data_{dataset_params}.npz")

        assert os.path.isfile(user_profile_parquet), f"Cannot find user_profile file {user_profile_parquet}. Please generate using data_process notebooks"
        assert os.path.isfile(ad_feature_parquet), f"Cannot find ad_feature file {ad_feature_parquet}. Please generate using data_process notebooks"
        assert os.path.isfile(train_file), f"Cannot find train data file {train_file}. Please generate using data_process notebooks"
        assert os.path.isfile(test_file), f"Cannot find test data file {test_file}. Please generate using data_process notebooks"

        self.mode = mode
        self.interaction_mapping = {-1: "ad_non_click" ,0: "browse", 1: "ad_click", 2: "favorite", 3: "add_to_cart", 4: "purchase"}

        self.user_feats = list(user_features)
        self.user_profile = pl.read_parquet(user_profile_parquet)
        self.user_encoder = PolarsOrdinalEncoder(fit_data=self.user_profile)

        self.ad_feats = list(ad_features)
        self.ad_feature = pl.read_parquet(ad_feature_parquet)
        self.ad_encoder = PolarsOrdinalEncoder(fit_data=self.ad_feature)

        self.input_dims = [self.user_encoder.feat_num_unique_with_null[feat] for feat in self.user_feats]
        self.output_dims = [self.ad_encoder.feat_num_unique_with_null[feat] for feat in self.ad_feats]

        self.transformed_ad_feature = self.ad_encoder.transform(self.ad_feature)
        self.ad_features = self.transformed_ad_feature.select(self.ad_feats).to_numpy()

        self.conditional_mappings = []
        for i in range(1, len(self.ad_feats)):
            conditional_map = (
                self.transformed_ad_feature
                .select(self.ad_feats[:i+1])
                .group_by(self.ad_feats[:i])
                .agg(
                    pl.col(self.ad_feats[i]).unique()
                )
                .select(*self.ad_feats[:i], pl.col(self.ad_feats[i]).list.set_difference([-1]))
                .to_pandas()
            )
            conditional_map.index = list(zip(*[conditional_map[self.ad_feats[j]] for j in range(i)]))
            self.conditional_mappings.append(conditional_map.to_dict()[self.ad_feats[i]])

        user_feat_map = {"user": 0, "gender": 1, "age": 2, "shopping": 3, "occupation": 4}
        user_feat_indices = [user_feat_map[feat] for feat in self.user_feats]

        ad_feat_map = {"adgroup": 0, "cate": 1, "brand": 2, "campaign": 3, "customer": 4}
        ad_feat_indices = [ad_feat_map[feat] for feat in self.ad_feats]

        if mode in ["pretrain", "finetune", "train"]:
            data = np.load(train_file)
        elif mode == "test":
            data = np.load(test_file)
        
        self.interaction_data = data["interaction_data"]
        train_idxs = np.full_like(self.interaction_data, True)
        if mode == "pretrain":
            train_idxs = (self.interaction_data != 1)
        elif mode == "finetune":
            train_idxs = (self.interaction_data == 1)

        self.user_data = data["user_data"][train_idxs, user_feat_indices]
        self.ads_data = data["ads_data"][train_idxs, ad_feat_indices]
        self.timestamps = data["timestamps"][train_idxs]
        self.interaction_data = data["interaction_data"][train_idxs]
        del data

    def __len__(self):
        return len(self.interaction_data)

    def __getitem__(self, idx):
        user_data, ads_data, timestamps, interactions = self.user_data[idx], self.ads_data[idx], self.timestamps[idx], self.interaction_data[idx]
        ads_masks = []
        for i, dim in enumerate(self.output_dims[1:]):
            mask_indices = self.conditional_mappings[i][tuple(ads_data[:i+1].tolist())]
            mask = np.ones(dim, dtype=bool)
            mask[mask_indices] = False
            ads_masks.append(mask)
        return user_data, ads_data, ads_masks, timestamps, interactions
