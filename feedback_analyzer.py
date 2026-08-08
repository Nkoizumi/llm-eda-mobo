# feedback_analyzer.py
#
# FeedbackAnalyzer — Analyzes EDA reports and applies corrective
# transformations to the DataFrame between feedback loop iterations.

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from typing import Optional


class FeedbackAnalyzer:
    """
    Analyzes EDA reports and applies corrective transformations
    to the DataFrame between feedback loop iterations.

    Two correction modes:
        1. suggest_corrections()   — Rule-based (original behaviour)
        2. apply_llm_corrections() — LLM-arbitrated corrections
                                     from ArbitratorLLM.decide()
    """

    def __init__(self, config):
        self.config     = config
        self.target_col = getattr(config, "target_col", None)
        # (transform, column) pairs already applied, so the feedback loop cannot
        # apply the same one twice. It fed its corrected frame back in as the
        # next iteration's input and nothing recorded what had been done, so a
        # column that stayed above the skew threshold after log1p was logged
        # again — log1p(log1p(x)) — for as many iterations as it kept failing.
        #
        # Measured on the bundled AmesHousing.csv (skew threshold 2.0): of six
        # flagged columns, four are STILL flagged after one log1p —
        #
        #     BsmtFin SF 2       4.14 -> 2.45
        #     Low Qual Fin SF   12.12 -> 8.58
        #     Bsmt Half Bath     3.94 -> 3.78
        #     Kitchen AbvGr      4.31 -> 3.53
        #
        # because their skew comes from a mass at zero, which a log cannot fix.
        # The negative-value guard cannot catch the repeat either: log1p output
        # is non-negative by construction.
        self._applied: set[tuple[str, str]] = set()

    def _unapplied(self, transform: str, cols: list) -> list:
        """Subset of `cols` this transform has not already been applied to."""
        fresh = [c for c in cols if (transform, c) not in self._applied]
        repeats = [c for c in cols if c not in fresh]
        if repeats and getattr(self.config, "verbose", False):
            print(f"   ↩️  {transform} already applied, skipping : {repeats}")
        return fresh

    def _mark(self, transform: str, cols: list) -> None:
        """Record columns this transform actually changed.

        Separate from `_unapplied` on purpose: a column that was filtered out
        downstream — log1p declining a column with negative values — has not
        been transformed, so marking it there would bar a legitimate later
        attempt once an earlier correction made it non-negative.
        """
        self._applied.update((transform, c) for c in cols)

    # ──────────────────────────────────────────────────────────────
    # Internal Guard
    # ──────────────────────────────────────────────────────────────

    def _safe_numeric_cols(
        self,
        df:        pd.DataFrame,
        candidates: list
    ) -> list:
        """
        ✅ Central safety filter used by ALL correction methods.

        From a list of candidate column names, returns only those that:
            1. Actually exist in the DataFrame
            2. Are NOT the target column
            3. Are numeric dtype

        This prevents the target column (e.g. 'RL') or any other
        string column from being passed to scalers / transformers.
        """
        target = self.target_col or getattr(self.config, "target_col", None)
        return [
            c for c in candidates
            if c in df.columns
            and c != target
            and pd.api.types.is_numeric_dtype(df[c])   # ← key guard
        ]

    # ──────────────────────────────────────────────────────────────
    # Mode 1 — Rule-Based Corrections
    # ──────────────────────────────────────────────────────────────

    def suggest_corrections(
        self,
        df,
        eda_report: dict,
        score:      float,
        iteration:  int
    ) -> pd.DataFrame:
        """
        Rule-based correction strategy.
        Applies heuristic transformations based on EDA report signals.

        Parameters
        ----------
        df         : Current DataFrame (may still have raw string target)
        eda_report : EDA report dict from AutoEDAPipeline
        score      : Current model score
        iteration  : Current iteration number

        Returns
        -------
        pd.DataFrame — corrected DataFrame
        """
        df     = df.copy()
        target = self.target_col or getattr(self.config, "target_col", None)

        if self.config.verbose:
            print(f"\n🔧 [FeedbackAnalyzer] Applying rule-based corrections "
                  f"(iteration {iteration})...")

        # ── 1. Drop ghost features (highly correlated) ────────────
        ghost_features = eda_report.get("ghost_features", [])
        if ghost_features:
            # ✅ Never drop target; check existence
            cols_to_drop = [
                c for c in ghost_features
                if c in df.columns and c != target
            ]
            if cols_to_drop:
                df.drop(columns=cols_to_drop, inplace=True)
                if self.config.verbose:
                    print(f"   🗑️  Dropped ghost features : {cols_to_drop}")

        # ── 2. RobustScaler on outlier-heavy features ─────────────
        outlier_features = eda_report.get("outlier_features", [])
        if outlier_features:
            # ✅ _safe_numeric_cols excludes target & non-numeric cols
            cols_to_scale = self._unapplied("robust",
                self._safe_numeric_cols(df, outlier_features))
            if cols_to_scale:
                scaler = RobustScaler()
                df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
                if self.config.verbose:
                    print(f"   📏 RobustScaler applied    : {cols_to_scale}")
                self._mark("robust", cols_to_scale)
            elif self.config.verbose and outlier_features:
                print(
                    f"   ⚠️  Outlier features skipped "
                    f"(non-numeric or target): {outlier_features}"
                )

        # ── 3. MinMaxScaler on damaged / sparse features ──────────
        damaged_features = eda_report.get("damaged_features", [])
        if damaged_features:
            # ✅ Same guard
            cols_to_restore = self._unapplied("minmax",
                self._safe_numeric_cols(df, damaged_features))
            if cols_to_restore:
                scaler = MinMaxScaler()
                df[cols_to_restore] = scaler.fit_transform(df[cols_to_restore])
                if self.config.verbose:
                    print(f"   🩹 MinMaxScaler applied    : {cols_to_restore}")
                self._mark("minmax", cols_to_restore)
            elif self.config.verbose and damaged_features:
                print(
                    f"   ⚠️  Damaged features skipped "
                    f"(non-numeric or target): {damaged_features}"
                )

        # ── 4. Log-transform skewed features ──────────────────────
        skewed_features = eda_report.get("skewed_features", [])
        if skewed_features:
            # ✅ Same guard + extra: must be non-negative for log1p
            cols_to_log = self._unapplied("log1p",
                self._safe_numeric_cols(df, skewed_features))
            applied, skipped = [], []

            for col in cols_to_log:
                if df[col].dropna().lt(0).any():
                    skipped.append(col)
                    continue
                df[col] = np.log1p(df[col])
                applied.append(col)

            self._mark("log1p", applied)
            if applied and self.config.verbose:
                print(f"   🔢 log1p applied           : {applied}")
            if skipped and self.config.verbose:
                print(f"   ⚠️  log1p skipped (negatives): {skipped}")

        return df

    # ──────────────────────────────────────────────────────────────
    # Mode 2 — LLM-Arbitrated Corrections
    # ──────────────────────────────────────────────────────────────

    def apply_llm_corrections(
        self,
        df,
        final_decision: dict
    ) -> pd.DataFrame:
        """
        Applies the arbitrated correction strategy produced by
        ArbitratorLLM.decide() to the DataFrame.

        Handles all 4 correction types:
            - drop_columns
            - rescale_columns      (RobustScaler or MinMaxScaler)
            - clip_columns         ([lower_pct, upper_pct])
            - log_transform_columns

        Parameters
        ----------
        df             : Current DataFrame
        final_decision : dict — output of ArbitratorLLM.decide()

        Returns
        -------
        pd.DataFrame — corrected DataFrame
        """
        df = df.copy()

        if self.config.verbose:
            print(
                f"\n🤖 [FeedbackAnalyzer] Applying LLM-arbitrated corrections..."
            )

        df = self._apply_drop(df,           final_decision)
        df = self._apply_rescale(df,        final_decision)
        df = self._apply_clip(df,           final_decision)
        df = self._apply_log_transform(df,  final_decision)

        return df

    # ──────────────────────────────────────────────────────────────
    # Private Helpers — LLM Correction Steps
    # ──────────────────────────────────────────────────────────────

    def _apply_drop(self, df: pd.DataFrame, decision: dict) -> pd.DataFrame:
        """
        Drops columns in drop_columns.
        Never drops the target column.
        """
        target = self.target_col or getattr(self.config, "target_col", None)
        cols_to_drop = [
            c for c in decision.get("drop_columns", [])
            if c in df.columns and c != target
        ]

        if not cols_to_drop:
            return df

        df.drop(columns=cols_to_drop, inplace=True)

        if self.config.verbose:
            print(f"   🗑️  Dropped columns        : {cols_to_drop}")

        return df

    def _apply_rescale(
        self, df: pd.DataFrame, decision: dict
    ) -> pd.DataFrame:
        """
        Applies RobustScaler or MinMaxScaler per column.

        Format: {"col_name": "RobustScaler"} or
                {"col_name": "MinMaxScaler"}
        """
        rescale_columns = decision.get("rescale_columns", {})

        if not rescale_columns:
            return df

        robust_cols = []
        minmax_cols = []

        for col, scaler_name in rescale_columns.items():
            # ✅ _safe_numeric_cols used inline per-column
            safe = self._safe_numeric_cols(df, [col])
            if not safe:
                if self.config.verbose:
                    print(
                        f"   ⚠️  Skipped rescale "
                        f"(non-numeric or target): {col}"
                    )
                continue
            if scaler_name == "MinMaxScaler":
                minmax_cols.append(col)
            else:
                robust_cols.append(col)

        if robust_cols:
            df[robust_cols] = RobustScaler().fit_transform(df[robust_cols])
            if self.config.verbose:
                print(f"   📏 RobustScaler applied    : {robust_cols}")

        if minmax_cols:
            df[minmax_cols] = MinMaxScaler().fit_transform(df[minmax_cols])
            if self.config.verbose:
                print(f"   📐 MinMaxScaler applied    : {minmax_cols}")

        return df

    def _apply_clip(
        self, df: pd.DataFrame, decision: dict
    ) -> pd.DataFrame:
        """
        Clips column values to [lower_pct, upper_pct] percentile bounds.
        Format: {"col_name": [1, 99]}
        """
        clip_columns = decision.get("clip_columns", {})

        if not clip_columns:
            return df

        for col, bounds in clip_columns.items():
            # ✅ Numeric + target guard
            safe = self._safe_numeric_cols(df, [col])
            if not safe:
                if self.config.verbose:
                    print(
                        f"   ⚠️  Skipped clip "
                        f"(non-numeric or target): {col}"
                    )
                continue
            if (
                not isinstance(bounds, (list, tuple))
                or len(bounds) != 2
                or not all(isinstance(b, (int, float)) for b in bounds)
            ):
                if self.config.verbose:
                    print(
                        f"   ⚠️  Skipped clip "
                        f"(invalid bounds {bounds}): {col}"
                    )
                continue

            lower_pct, upper_pct = bounds
            lower_val = np.percentile(df[col].dropna(), lower_pct)
            upper_val = np.percentile(df[col].dropna(), upper_pct)
            df[col]   = df[col].clip(lower=lower_val, upper=upper_val)

            if self.config.verbose:
                print(
                    f"   ✂️  Clipped [{lower_pct}–{upper_pct}pct]  : "
                    f"{col} → [{lower_val:.4f}, {upper_val:.4f}]"
                )

        return df

    def _apply_log_transform(
        self, df: pd.DataFrame, decision: dict
    ) -> pd.DataFrame:
        """
        Applies log1p to specified columns.
        Skips non-numeric, target, and columns with negative values.
        """
        log_cols = decision.get("log_transform_columns", [])

        if not log_cols:
            return df

        # ✅ Numeric + target guard first
        safe_cols = self._unapplied("log1p", self._safe_numeric_cols(df, log_cols))
        applied, skipped = [], []

        for col in safe_cols:
            if df[col].dropna().lt(0).any():
                skipped.append(col)
                if self.config.verbose:
                    print(
                        f"   ⚠️  Skipped log1p "
                        f"(negative values): {col}"
                    )
                continue
            df[col] = np.log1p(df[col])
            applied.append(col)

        # Columns that didn't pass the numeric guard
        fully_skipped = [c for c in log_cols if c not in safe_cols]
        skipped.extend(fully_skipped)

        self._mark("log1p", applied)
        if applied and self.config.verbose:
            print(f"   🔢 log1p applied           : {applied}")
        if fully_skipped and self.config.verbose:
            print(
                f"   ⚠️  log1p skipped "
                f"(non-numeric or target): {fully_skipped}"
            )

        return df
