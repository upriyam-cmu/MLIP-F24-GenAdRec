import numpy as np
from sklearn import preprocessing as P


def _new_arr(*shape):
    # return np.full(shape, -1)
    return np.empty(shape, dtype=np.int64)


class OrdinalEncoder:
    def __init__(self):
        self.encoder = P.OrdinalEncoder(dtype=int)
    
    def fit(self, X):
        self.encoder.fit(X.reshape(-1, 1))
        return self
    
    def transform(self, X):
        return self.encoder.transform(X.reshape(-1, 1)).flatten()


def preprocess(user_data, ad_data, behavior_data, interaction_data, *, inplace=False, time_stamp_block_size=900):
    if not inplace:
        user_data = user_data.copy()
        ad_data = ad_data.copy()
        behavior_data = behavior_data.copy()
        interaction_data = interaction_data.copy()
    
    # drop pid
    interaction_data.drop(columns='pid', inplace=True)

    # block timestamps & sort
    behavior_data.time_stamp //= time_stamp_block_size
    interaction_data.time_stamp //= time_stamp_block_size

    behavior_data.sort_values('time_stamp', inplace=True)
    interaction_data.sort_values('time_stamp', inplace=True)

    # remove duplicates
    behavior_data.drop_duplicates(inplace=True)
    interaction_data.drop_duplicates(inplace=True)

    # sanitize NaNs
    ad_data.brand = ad_data.brand.apply(lambda x: 0 if np.isnan(x) else round(x))

    # set up ordinal encoders (we don't care about anything other than these four)
    ad_categories = np.unique(np.concatenate([ad_data.cate_id.unique(), behavior_data.cate.unique()]))
    ad_brands = np.unique(np.concatenate([ad_data.brand.unique(), behavior_data.brand.unique()]))
    ad_customers = ad_data.customer.unique()
    ad_campaigns = ad_data.campaign_id.unique()

    ad_categories_enc = OrdinalEncoder().fit(ad_categories)
    ad_brands_enc = OrdinalEncoder().fit(ad_brands)
    ad_customers_enc = OrdinalEncoder().fit(ad_customers)
    ad_campaigns_enc = OrdinalEncoder().fit(ad_campaigns)

    # build data arrays
    ## user data -- not used
    user_data_np = _new_arr(len(user_data), 0)

    ## ad data -- [ad_id, category, brand, customer, campaign]
    ad_data_np = _new_arr(len(ad_data), 5)
    ad_data_np[:, 0] = ad_data.adgroup_id.to_numpy()
    ad_data_np[:, 1] = ad_categories_enc.transform(ad_data.cate_id.to_numpy())
    ad_data_np[:, 2] = ad_brands_enc.transform(ad_data.brand.to_numpy())
    ad_data_np[:, 3] = ad_customers_enc.transform(ad_data.customer.to_numpy())
    ad_data_np[:, 4] = ad_campaigns_enc.transform(ad_data.campaign_id.to_numpy())

    ## behavior data -- [user_id, time_stamp, category, brand]
    behavior_data_np = _new_arr(len(behavior_data), 4)
    behavior_data_np[:, 0] = behavior_data.user.to_numpy()
    behavior_data_np[:, 1] = behavior_data.time_stamp.to_numpy()
    behavior_data_np[:, 2] = ad_categories_enc.transform(behavior_data.cate.to_numpy())
    behavior_data_np[:, 3] = ad_brands_enc.transform(behavior_data.brand.to_numpy())

    ## interaction data -- [user_id, time_stamp, ad_id, click]
    interaction_data_np = _new_arr(len(interaction_data), 4)
    interaction_data_np[:, 0] = interaction_data.user.to_numpy()
    interaction_data_np[:, 1] = interaction_data.time_stamp.to_numpy()
    interaction_data_np[:, 2] = interaction_data.adgroup_id.to_numpy()
    interaction_data_np[:, 3] = interaction_data.clk.to_numpy()

    # return data arrays
    return user_data_np, ad_data_np, behavior_data_np, interaction_data_np
