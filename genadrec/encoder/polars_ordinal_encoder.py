import polars as pl

class PolarsOrdinalEncoder:
    def __init__(self, fit_data: pl.DataFrame):
        self.mappings = {}
        self.inverse_mappings = {}
        self.feat_num_unique_with_null = {}
        self.feat_has_null = {}
        for feat in fit_data.columns:
            feat_unique = fit_data[feat].unique().sort(nulls_last=True)
            self.feat_has_null[feat] = feat_unique.has_nulls()
            feat_encodings = (feat_unique.rank("ordinal") - 1).cast(pl.Int32).replace(None, -1)
            self.mappings[feat] = {"old": feat_unique, "new": feat_encodings}
            self.inverse_mappings[feat] = {"old": feat_encodings, "new": feat_unique}
            self.feat_num_unique_with_null[feat] = len(feat_encodings)

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.select(pl.col(feat).replace_strict(**self.mappings[feat]) if feat in self.mappings else pl.col(feat) for feat in data.columns)

    def inverse_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.select(pl.col(feat).replace_strict(**self.inverse_mappings[feat]) if feat in self.inverse_mappings else pl.col(feat) for feat in data.columns)