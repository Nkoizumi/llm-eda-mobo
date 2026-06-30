# pipeline/shap_feedback.py

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class SHAPFeedbackResult:
    """Structured output from one SHAP feedback pass."""

    iteration:            int
    top_features:         list[str]         = field(default_factory=list)
    ghost_features:       list[str]         = field(default_factory=list)
    damaged_features:     list[str]         = field(default_factory=list)
    recommended_drops:    list[str]         = field(default_factory=list)
    shap_values_df:       pd.DataFrame      = field(default_factory=pd.DataFrame)
    mean_abs_shap:        pd.Series         = field(default_factory=pd.Series)
    correction_applied:   str               = "none"


class SHAPFeedbackAnalyzer:
    """
    Uses SHAP values to audit each iteration of the EDA feedback loop.

    Detects:
      1. Ghost features  — high importance but collinear with another feature
      2. Damaged features — importance collapsed to near-zero after transform
      3. Outlier distortion — SHAP variance spiked on few rows
    """

    def __init__(
        self,
        task_type:              str   = "classification",
        ghost_corr_threshold:   float = 0.90,
        damaged_shap_threshold: float = 0.001,
        spike_percentile:       float = 99.0,
        spike_ratio:            float = 10.0,
        random_state:           int   = 42,
        n_estimators:           int   = 50,
        verbose:                bool  = True,
    ):
        self.task_type              = task_type
        self.ghost_corr_threshold   = ghost_corr_threshold
        self.damaged_shap_threshold = damaged_shap_threshold
        self.spike_percentile       = spike_percentile
        self.spike_ratio            = spike_ratio
        self.random_state           = random_state
        self.n_estimators           = n_estimators
        self.verbose                = verbose

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ──────────────────────────────────────────────────────────────────────
    def analyze(
        self,
        df:           pd.DataFrame,
        target_col:   str,
        iteration:    int = 1,
        prev_shap:    pd.Series | None = None,  # mean abs SHAP from prev iter
    ) -> SHAPFeedbackResult:
        """
        Run SHAP analysis on current iteration's DataFrame.

        Args:
            df          : Processed DataFrame (features + target).
            target_col  : Name of target column.
            iteration   : Current loop iteration number.
            prev_shap   : Mean absolute SHAP Series from previous iteration.
                          Used to detect damaged features.

        Returns:
            SHAPFeedbackResult with diagnostics and recommendations.
        """
        import shap
        from xgboost import XGBClassifier, XGBRegressor

        result = SHAPFeedbackResult(iteration=iteration)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # ── Fit a fast XGBoost just for SHAP ──────────────────────────────
        model = (
            XGBClassifier(
                n_estimators=self.n_estimators,
                verbosity=0,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=self.random_state,
            )
            if self.task_type == "classification"
            else XGBRegressor(
                n_estimators=self.n_estimators,
                verbosity=0,
                random_state=self.random_state,
            )
        )
        model.fit(X, y)

        # ── Compute SHAP values ───────────────────────────────────────────
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # For multiclass XGBoost, shap_values is a list — take mean across classes
        if isinstance(shap_values, list):
            shap_arr = np.mean(np.abs(shap_values), axis=0)
        else:
            shap_arr = np.abs(shap_values)

        shap_df          = pd.DataFrame(shap_arr, columns=X.columns)
        mean_abs_shap    = shap_df.mean().sort_values(ascending=False)

        result.shap_values_df = shap_df
        result.mean_abs_shap  = mean_abs_shap
        result.top_features   = mean_abs_shap.head(10).index.tolist()

        if self.verbose:
            print(f"\n  [SHAP iter={iteration}] Top 5 features by SHAP:")
            for feat, val in mean_abs_shap.head(5).items():
                print(f"    {feat:<30} {val:.5f}")

        # ── Audit 1: Ghost Features ───────────────────────────────────────
        result.ghost_features = self._detect_ghost_features(
            X, mean_abs_shap
        )

        # ── Audit 2: Damaged Features ─────────────────────────────────────
        if prev_shap is not None:
            result.damaged_features = self._detect_damaged_features(
                mean_abs_shap, prev_shap
            )

        # ── Audit 3: Outlier Distortion ───────────────────────────────────
        distorted = self._detect_outlier_distortion(shap_df)
        if distorted:
            result.correction_applied = "outlier_distortion_detected"
            if self.verbose:
                print(
                    f"  [SHAP iter={iteration}] ⚠️  Outlier distortion in: "
                    f"{distorted}"
                )

        # ── Build recommended drops ───────────────────────────────────────
        result.recommended_drops = list(
            set(result.ghost_features + result.damaged_features)
        )

        if result.recommended_drops and self.verbose:
            print(
                f"  [SHAP iter={iteration}] 🗑️  Recommended drops: "
                f"{result.recommended_drops}"
            )

        return result

    # ──────────────────────────────────────────────────────────────────────
    # PRIVATE DIAGNOSTICS
    # ──────────────────────────────────────────────────────────────────────
    def _detect_ghost_features(
        self,
        X:             pd.DataFrame,
        mean_abs_shap: pd.Series,
    ) -> list[str]:
        """
        Ghost feature = high SHAP impact AND highly correlated
        with a higher-ranked feature (likely a near-duplicate).
        """
        num_X       = X.select_dtypes(include="number")
        corr_matrix = num_X.corr().abs()
        ranked_cols = mean_abs_shap.index.tolist()
        ghosts      = []

        for i, col in enumerate(ranked_cols):
            if col not in corr_matrix.columns:
                continue
            # Check all higher-ranked features
            for higher_col in ranked_cols[:i]:
                if higher_col not in corr_matrix.columns:
                    continue
                if corr_matrix.loc[col, higher_col] >= self.ghost_corr_threshold:
                    ghosts.append(col)
                    if self.verbose:
                        print(
                            f"  [SHAP] 👻 Ghost: '{col}' corr={corr_matrix.loc[col, higher_col]:.3f} "
                            f"with '{higher_col}'"
                        )
                    break  # one reason is enough to flag it

        return ghosts

    def _detect_damaged_features(
        self,
        curr_shap: pd.Series,
        prev_shap: pd.Series,
    ) -> list[str]:
        """
        Damaged feature = SHAP dropped by > 80% vs previous iteration.
        Suggests a transform destroyed the signal in that feature.
        """
        shared   = curr_shap.index.intersection(prev_shap.index)
        damaged  = []

        for feat in shared:
            prev_val = prev_shap[feat]
            curr_val = curr_shap[feat]

            if prev_val < self.damaged_shap_threshold:
                continue  # was already negligible — not "damaged"

            drop_ratio = (prev_val - curr_val) / (prev_val + 1e-9)

            if drop_ratio > 0.80 and curr_val < self.damaged_shap_threshold:
                damaged.append(feat)
                if self.verbose:
                    print(
                        f"  [SHAP] 💥 Damaged: '{feat}' SHAP "
                        f"{prev_val:.5f} → {curr_val:.5f} "
                        f"(dropped {drop_ratio * 100:.1f}%)"
                    )

        return damaged

    def _detect_outlier_distortion(
        self,
        shap_df: pd.DataFrame,
    ) -> list[str]:
        """
        Outlier distortion = a small number of rows have disproportionately
        large SHAP values (spike ratio > threshold).
        """
        distorted = []

        for col in shap_df.columns:
            vals     = shap_df[col].values
            p99      = np.percentile(np.abs(vals), self.spike_percentile)
            median   = np.median(np.abs(vals))

            if median < 1e-6:
                continue  # near-zero feature — skip

            ratio = p99 / (median + 1e-9)
            if ratio > self.spike_ratio:
                distorted.append(col)

        return distorted
