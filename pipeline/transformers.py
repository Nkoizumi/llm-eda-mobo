import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    PowerTransformer, OneHotEncoder,
    OrdinalEncoder, PolynomialFeatures
)
from sklearn.ensemble import IsolationForest

# ─────────────────────────────────────────────────────────────────────────────
# 1. SmartImputer
# ─────────────────────────────────────────────────────────────────────────────
class SmartImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="median", knn_k=5):
        self.strategy = strategy
        self.knn_k = knn_k
        self.imputer_ = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        self.feature_names_ = df.columns.tolist()
        if self.strategy in ("mean", "median"):
            self.imputer_ = SimpleImputer(strategy=self.strategy)
        elif self.strategy == "knn":
            self.imputer_ = KNNImputer(n_neighbors=self.knn_k)
        elif self.strategy == "missforest":
            try:
                from missforest import MissForest
                self.imputer_ = MissForest()
            except ImportError:
                self.imputer_ = KNNImputer(n_neighbors=self.knn_k)
        elif self.strategy == "iterative":
            try:
                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                from sklearn.impute import IterativeImputer
                self.imputer_ = IterativeImputer(random_state=0, max_iter=10)
            except ImportError:
                self.imputer_ = KNNImputer(n_neighbors=self.knn_k)
        else:
            # "none" or any unrecognised strategy — use median as a safe default
            self.imputer_ = SimpleImputer(strategy="median")
        self.imputer_.fit(df)
        return self

    def transform(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        result = self.imputer_.transform(df)
        return pd.DataFrame(result, columns=self.feature_names_, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SkewnessKurtosisCorrector
# ─────────────────────────────────────────────────────────────────────────────
class SkewnessKurtosisCorrector(BaseEstimator, TransformerMixin):
    SKEW_THRESHOLD = 0.5
    KURT_THRESHOLD = 3.0

    def __init__(self, method="yeo-johnson"):
        self.method = method
        self.transformers_ = {}
        self.cols_to_transform_ = []
        self.before_stats_ = {}
        self.after_stats_ = {}

    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        for col in numeric_cols:
            skew = abs(df[col].skew())
            kurt = abs(df[col].kurtosis())
            self.before_stats_[col] = {"skew": round(df[col].skew(), 3),
                                        "kurt": round(df[col].kurtosis(), 3)}
            if skew > self.SKEW_THRESHOLD or kurt > self.KURT_THRESHOLD:
                self.cols_to_transform_.append(col)
                pt = PowerTransformer(method=self.method, standardize=False)
                pt.fit(df[[col]])
                self.transformers_[col] = pt
        return self

    def transform(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        for col, pt in self.transformers_.items():
            if col in df.columns:
                df[col] = pt.transform(df[[col]])
                self.after_stats_[col] = {
                    "skew": round(df[col].skew(), 3),
                    "kurt": round(df[col].kurtosis(), 3)
                }
        return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. OutlierHandler
# ─────────────────────────────────────────────────────────────────────────────
class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method="iqr", threshold=1.5):
        self.method = method
        self.threshold = threshold
        self.bounds_ = {}
        self.iso_forest_ = None
        self.outlier_counts_ = {}

    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if self.method == "iqr":
            for col in numeric_cols:
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                lb, ub = Q1 - self.threshold * IQR, Q3 + self.threshold * IQR
                self.bounds_[col] = (lb, ub)
                self.outlier_counts_[col] = int(((df[col] < lb) | (df[col] > ub)).sum())
        elif self.method == "zscore":
            for col in numeric_cols:
                mean, std = df[col].mean(), df[col].std()
                lb, ub = mean - self.threshold * std, mean + self.threshold * std
                self.bounds_[col] = (lb, ub)
                self.outlier_counts_[col] = int(((df[col] < lb) | (df[col] > ub)).sum())
        elif self.method == "isolation_forest":
            self.iso_forest_ = IsolationForest(contamination=0.05, random_state=42)
            self.iso_forest_.fit(df[numeric_cols])
        return self

    def transform(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if self.method in ("iqr", "zscore"):
            for col in numeric_cols:
                if col in self.bounds_:
                    lb, ub = self.bounds_[col]
                    df[col] = df[col].clip(lower=lb, upper=ub)
        elif self.method == "isolation_forest" and self.iso_forest_ is not None:
            preds = self.iso_forest_.predict(df[numeric_cols])
            # Clip outlier rows to column medians rather than setting NaN,
            # which would re-introduce missing values after the imputer step.
            if (preds == -1).any():
                medians = df[numeric_cols].median()
                df.loc[preds == -1, numeric_cols] = medians.values
        return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. HighCorrelationRemover
# ─────────────────────────────────────────────────────────────────────────────
class HighCorrelationRemover(BaseEstimator, TransformerMixin):
    """Drop columns that are |Pearson r| > threshold with an earlier column.

    Position-based: ``fit`` records the integer positions to drop based on the
    column ORDER seen at fit time. ``transform`` then drops those positions
    regardless of input type (numpy array or DataFrame), so a sklearn Pipeline
    can call this step on either with consistent results. Implements
    ``get_feature_names_out`` so downstream Pipeline steps (and
    ``Pipeline.get_feature_names_out``) work transparently.
    """

    def __init__(self, threshold=0.90):
        self.threshold = threshold

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            data = X.to_numpy(dtype=float, copy=False, na_value=np.nan)
        else:
            X_arr = np.asarray(X, dtype=float)
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X_arr.shape[1])], dtype=object)
            data = X_arr

        n = data.shape[1]
        if n < 2:
            self.features_to_drop_positions_ = np.array([], dtype=int)
            self.corr_matrix_ = np.zeros((n, n))
            return self

        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.abs(np.corrcoef(data, rowvar=False))
        # Constant columns yield NaN correlations — treat as uncorrelated.
        corr = np.nan_to_num(corr, nan=0.0)

        # For each pair (i, j) with i < j, drop j if |corr| > threshold.
        # Equivalent to the original "any(upper[col] > threshold)" rule.
        i_idx, j_idx = np.triu_indices(n, k=1)
        high = corr[i_idx, j_idx] > self.threshold
        drop_positions = np.unique(j_idx[high])

        self.features_to_drop_positions_ = drop_positions.astype(int)
        self.corr_matrix_ = corr
        return self

    def _keep_mask(self) -> np.ndarray:
        n = len(self.feature_names_in_)
        keep = np.ones(n, dtype=bool)
        keep[self.features_to_drop_positions_] = False
        return keep

    def transform(self, X):
        keep = self._keep_mask()
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, keep]
        return np.asarray(X)[:, keep]

    def get_feature_names_out(self, input_features=None):
        names = (np.asarray(input_features, dtype=object)
                 if input_features is not None
                 else self.feature_names_in_)
        return names[self._keep_mask()]

    # Backward-compat shim for any caller that read the old attribute.
    @property
    def features_to_drop_(self):
        return list(self.feature_names_in_[self.features_to_drop_positions_])


# ─────────────────────────────────────────────────────────────────────────────
# 5. SmartFeatureEngineer
# ─────────────────────────────────────────────────────────────────────────────
class SmartFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, add_polynomial=False, degree=2,
                 interactions_only=False, add_interactions=False):
        self.add_polynomial = add_polynomial
        self.degree = degree
        self.interactions_only = interactions_only
        self.add_interactions = add_interactions
        self.poly_ = None
        self.numeric_cols_ = []
        self.new_feature_names_ = []

    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        self.numeric_cols_ = df.select_dtypes(include=np.number).columns.tolist()
        if (self.add_polynomial or self.add_interactions) and self.numeric_cols_:
            self.poly_ = PolynomialFeatures(
                degree=self.degree,
                interaction_only=self.interactions_only,
                include_bias=False
            )
            self.poly_.fit(df[self.numeric_cols_])
            poly_names = self.poly_.get_feature_names_out(self.numeric_cols_)
            self.new_feature_names_ = [n for n in poly_names if n not in self.numeric_cols_]
        return self

    def transform(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        if self.poly_ is not None and self.numeric_cols_:
            poly_arr = self.poly_.transform(df[self.numeric_cols_])
            poly_names = self.poly_.get_feature_names_out(self.numeric_cols_)
            new_idx = [i for i, n in enumerate(poly_names) if n not in self.numeric_cols_]
            new_df = pd.DataFrame(
                poly_arr[:, new_idx], columns=self.new_feature_names_, index=df.index
            )
            df = pd.concat([df, new_df], axis=1)
        return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. SmartCategoryEncoder
# ─────────────────────────────────────────────────────────────────────────────
class SmartCategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, onehot_cols=None, ordinal_cols=None):
        self.onehot_cols = onehot_cols or []
        self.ordinal_cols = ordinal_cols or []
        self.ohe_ = None
        self.oe_ = None
        self.ohe_feature_names_ = []

    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        valid_ohe = [c for c in self.onehot_cols if c in df.columns]
        valid_ord = [c for c in self.ordinal_cols if c in df.columns]
        if valid_ohe:
            self.ohe_ = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore", drop="first"
            )
            self.ohe_.fit(df[valid_ohe])
            self.ohe_feature_names_ = self.ohe_.get_feature_names_out(valid_ohe).tolist()
        if valid_ord:
            self.oe_ = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            self.oe_.fit(df[valid_ord])
        return self

    def transform(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        valid_ohe = [c for c in self.onehot_cols if c in df.columns]
        valid_ord = [c for c in self.ordinal_cols if c in df.columns]
        if self.ohe_ and valid_ohe:
            ohe_arr = self.ohe_.transform(df[valid_ohe])
            ohe_df = pd.DataFrame(ohe_arr, columns=self.ohe_feature_names_, index=df.index)
            df = df.drop(columns=valid_ohe)
            df = pd.concat([df, ohe_df], axis=1)
        if self.oe_ and valid_ord:
            df[valid_ord] = self.oe_.transform(df[valid_ord])
        return df


from sklearn.pipeline import Pipeline

class Transformer:
    """Unified wrapper combining all individual transformers."""

    def __init__(self):
        self.pipeline = Pipeline([
            ("imputer",       SmartImputer()),
            ("skewness",      SkewnessKurtosisCorrector()),
            ("outlier",       OutlierHandler()),
            ("corr_remover",  HighCorrelationRemover()),
            ("feat_engineer", SmartFeatureEngineer()),
            ("encoder",       SmartCategoryEncoder()),
        ])

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None):
        return self.pipeline.fit_transform(X, y)
