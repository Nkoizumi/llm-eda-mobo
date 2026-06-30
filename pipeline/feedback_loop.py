# pipeline/feedback_loop.py

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from pipeline.orchestrator import AutoEDAPipeline, Orchestrator
from feedback_analyzer     import FeedbackAnalyzer
from feedback_controller   import FeedbackConfig
from pipeline.shap_feedback import SHAPFeedbackAnalyzer


# ─────────────────────────────────────────────────────────────────────────────
class ModelTrainerAdapter:
    """
    Wraps Orchestrator to provide a clean train_and_evaluate() interface.
    """

    def __init__(self, config: FeedbackConfig):
        self.config = config

    def train_and_evaluate(self, processed_df: pd.DataFrame):
        orchestrator = Orchestrator(
            task         = self.config.task_type,
            n_splits     = 5,
            random_state = 42,
        )

        pipeline_output = orchestrator.run(
            processed_df,
            target_col      = self.config.target_col,
            pre_transformed = True,
        )

        # ✅ Extract score from comparison_df — guaranteed to have correct values
        comparison_df = pipeline_output["comparison"]

        print(f"  [DEBUG] comparison_df columns : {comparison_df.columns.tolist()}")
        print(f"  [DEBUG] comparison_df head    :\n{comparison_df.to_string()}")

        if self.config.task_type == "regression":
            score_col = "R2"        # ← adjust to match your compare_regression_results output
        else:
            score_col = "Accuracy"  # ← matches the FINAL MODEL COMPARISON table above ✅

        # ── Guard: score column must exist ────────────────────────────
        if score_col not in comparison_df.columns:
            available = comparison_df.columns.tolist()
            raise KeyError(
                f"[ModelTrainerAdapter] score_col '{score_col}' not found "
                f"in comparison_df. Available: {available}"
            )

        # ── Pick the best row (Rank index = 1 is best) ────────────────
        best_row   = comparison_df.iloc[0]          # ✅ Rank 1 = top row
        best_model = best_row["Model"]
        best_score = float(best_row[score_col])

        print(f"  [DEBUG] Best model : {best_model}")
        print(f"  [DEBUG] Best score : {best_score}")

        return best_score, best_model


# ─────────────────────────────────────────────────────────────────────────────
class FeedbackLoop:

    def __init__(self, config: FeedbackConfig = None):
        self.config     = config or FeedbackConfig()
        self.iteration  = 0
        self.history    = []
        self.best_score = -np.inf
        self.best_df    = None

        self.eda = AutoEDAPipeline(
            target_col    = self.config.target_col
                            if hasattr(self.config, "target_col") else "target",
            task          = self.config.task_type,
            use_local_llm = False,
        )
        self.trainer  = ModelTrainerAdapter(self.config)
        self.analyzer = FeedbackAnalyzer(self.config)

        # ✅ NEW — SHAP feedback components
        self.shap_analyzer = SHAPFeedbackAnalyzer(
            task_type   = self.config.task_type,
            verbose     = self.config.verbose,
        )
        self.prev_shap: pd.Series | None = None

    # ─────────────────────────────────────────────────────────────────────
    def run(self, raw_df: pd.DataFrame):

        print("\n" + "█" * 60)
        print("  [FEEDBACK LOOP] run() called")
        print("█" * 60)

        # ── Guards (unchanged) ─────────────────────────────────────────
        if raw_df is None:
            print("  [FATAL] raw_df is None — aborting.")
            return None, None, []

        if not isinstance(raw_df, pd.DataFrame):
            print(f"  [FATAL] raw_df is type {type(raw_df)}, expected DataFrame.")
            return None, None, []

        if raw_df.empty:
            print("  [FATAL] raw_df is empty — aborting.")
            return None, None, []

        print(f"  [OK] DataFrame shape      : {raw_df.shape}")
        print(f"  [OK] DataFrame columns    : {raw_df.columns.tolist()}")
        print(f"  [OK] Target column        : {self.eda.target_col}")
        print(f"  [OK] Task type            : {self.config.task_type}")
        print(f"  [OK] Max iterations       : {self.config.max_iterations}")
        print(f"  [OK] Metric threshold     : {self.config.metric_threshold}")

        if self.eda.target_col not in raw_df.columns:
            print(
                f"  [FATAL] target_col '{self.eda.target_col}' "
                f"NOT found in DataFrame columns!"
            )
            print(f"  Available columns: {raw_df.columns.tolist()}")
            return None, None, []

        print(f"  [OK] target_col found in DataFrame ✓")
        print("█" * 60 + "\n")

        df = raw_df.copy()

        while self.iteration < self.config.max_iterations:
            self.iteration += 1
            print(f"\n  [LOOP] Starting iteration {self.iteration} ...")
            self._log(f"\n{'='*55}")
            self._log(f"  Feedback Iteration #{self.iteration}")
            self._log(f"{'='*55}")

            try:
                # ── Step 1: Separate target, build & apply EDA pipeline ───
                print(f"  [STEP 1] Separating target col and running EDA pipeline...")

                y_series = df[self.eda.target_col]
                X_only   = df.drop(columns=[self.eda.target_col], errors="ignore")
                print(f"  [STEP 1] X_only shape  : {X_only.shape}")
                print(f"  [STEP 1] y_series shape: {y_series.shape}")

                self.eda.build_pipeline(X_only)
                print(f"  [STEP 1] build_pipeline() completed ✓")

                X_transformed = self.eda.get_transformed_df(X_only)
                print(f"  [STEP 1] get_transformed_df() completed ✓")
                print(f"  [STEP 1] X_transformed shape: {X_transformed.shape}")

                processed_df = X_transformed.copy()

                y_clean = y_series.reset_index(drop=True)
                if pd.api.types.is_float_dtype(y_clean):
                    if (y_clean == y_clean.round()).all():
                        y_clean = y_clean.astype(int)
                        print(
                            f"  [STEP 1] y cast float → int ✅ "
                            f"unique: {sorted(y_clean.unique())[:10]}"
                        )

                processed_df[self.eda.target_col] = y_clean.values

                print(f"  [STEP 1] processed_df shape   : {processed_df.shape}")
                print(f"  [STEP 1] y dtype in processed : "
                      f"{processed_df[self.eda.target_col].dtype}")
                print(f"  [STEP 1] target_col present   : "
                      f"{self.eda.target_col in processed_df.columns} ✓")

                # ── Step 2: Build EDA report ───────────────────────────────
                print(f"  [STEP 2] Building EDA report...")
                eda_report = self._build_eda_report(df)
                print(f"  [STEP 2] EDA report keys: {list(eda_report.keys())} ✓")

                # ✅ ─────────────────────────────────────────────────────────
                # ── Step 2b: SHAP Audit ───────────────────────────────────
                # ─────────────────────────────────────────────────────────
                print(f"  [STEP 2b] Running SHAP audit...")
                shap_result = self.shap_analyzer.analyze(
                    df         = processed_df,
                    target_col = self.eda.target_col,
                    iteration  = self.iteration,
                    prev_shap  = self.prev_shap,        # None on first iter
                )

                # ── Drop SHAP-flagged columns before scoring ──────────────
                if shap_result.recommended_drops:
                    cols_before = processed_df.shape[1]
                    processed_df = processed_df.drop(
                        columns = shap_result.recommended_drops,
                        errors  = "ignore",
                    )
                    cols_after = processed_df.shape[1]
                    print(
                        f"  [STEP 2b] SHAP dropped "
                        f"{cols_before - cols_after} column(s): "
                        f"{shap_result.recommended_drops}"
                    )
                else:
                    print(f"  [STEP 2b] SHAP audit clean — no drops needed ✓")

                # ── Save SHAP state for next iteration ────────────────────
                self.prev_shap = shap_result.mean_abs_shap
                print(
                    f"  [STEP 2b] Top SHAP features: "
                    f"{shap_result.top_features[:5]} ✓"
                )

                # Enrich eda_report with SHAP diagnostic fields so that
                # FeedbackAnalyzer.suggest_corrections() can act on them.
                eda_report["ghost_features"]   = shap_result.ghost_features
                eda_report["damaged_features"] = shap_result.damaged_features
                eda_report["outlier_features"] = getattr(shap_result, "outlier_features", [])
                # ✅ ─────────────────────────────────────────────────────────

                # ── Step 3: Train & evaluate ───────────────────────────────
                print(f"  [STEP 3] Training and evaluating model...")
                score, model_name = self.trainer.train_and_evaluate(processed_df)
                print(f"  [STEP 3] Score: {score:.4f} | Model: {model_name} ✓")

            except Exception as e:
                import traceback
                print(f"\n  [!!!] ITERATION {self.iteration} FAILED")
                print(f"  [!!!] Error type : {type(e).__name__}")
                print(f"  [!!!] Error msg  : {e}")
                print(f"  [!!!] Full traceback:")
                traceback.print_exc()
                print(f"  [!!!] Breaking out of loop.\n")
                break

            # ── Step 4: Log history ────────────────────────────────────────
            correction = self._correction_label(self.iteration - 1)
            self.history.append({
                "iteration":   self.iteration,
                "model":       model_name if model_name else "N/A",
                "score":       float(score),
                "corrections": correction,
                # ✅ NEW SHAP fields
                "shap_top5":   shap_result.top_features[:5],
                "ghosts":      shap_result.ghost_features,
                "damaged":     shap_result.damaged_features,
            })
            print(f"  [STEP 4] Appended to history ✓  (total: {len(self.history)})")

            metric = "R²" if self.config.task_type == "regression" else "Accuracy"
            self._log(f"  {metric}: {score:.4f} | Model: {model_name}")

            # ── Step 5: Check threshold ────────────────────────────────────
            if score >= self.config.metric_threshold:
                print(
                    f"  [DONE] Threshold met! Score {score:.4f} ≥ "
                    f"{self.config.metric_threshold}"
                )
                self._log(
                    f"\n  Threshold met at iteration {self.iteration}!"
                    f" Score: {score:.4f}"
                )
                return processed_df, model_name, self.history

            # ── Step 6: Track best ─────────────────────────────────────────
            if score > self.best_score:
                self.best_score = score
                self.best_df    = processed_df.copy()
                print(f"  [STEP 6] New best score: {self.best_score:.4f} ✓")

            # ── Step 7: Apply corrections ──────────────────────────────────
            print(f"  [STEP 7] Applying correction: '{correction}'...")
            df = self.analyzer.suggest_corrections(
                df, eda_report, score, self.iteration
            )
            print(f"  [STEP 7] Corrections applied. New df shape: {df.shape} ✓")

        print(f"\n  [DONE] Loop finished. Best Score: {self.best_score:.4f}")
        print(f"  [DONE] Total history entries   : {len(self.history)}\n")
        self._log(
            f"\n  Max iterations reached."
            f" Best Score: {self.best_score:.4f}"
        )
        return self.best_df, None, self.history

    # ─────────────────────────────────────────────────────────────────────
    def print_history(self):
        metric = "R²" if self.config.task_type == "regression" \
                 else "Accuracy"
        print(f"\n{'='*55}")
        print(f"  FEEDBACK LOOP HISTORY")
        print(f"{'='*55}")
        for h in self.history:
            passed = h["score"] >= self.config.metric_threshold
            flag   = "PASS" if passed else "FAIL"
            print(
                f"  [{flag}] Iter {h['iteration']} | "
                f"{h['model']:<18} | "
                f"{metric}: {h['score']:.4f} | "
                f"Fix: {h['corrections']}"
            )
        print(f"\n  Best Score : {self.best_score:.4f}")
        print(f"  Threshold  : {self.config.metric_threshold}")
        print(f"{'='*55}\n")

    # ─────────────────────────────────────────────────────────────────────
    def _build_eda_report(self, df: pd.DataFrame) -> dict:
        """
        Builds a minimal EDA report dict that FeedbackAnalyzer expects.
        Mirrors what AutoEDAPipeline would normally pass.
        """
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        missing = {
            col: int(df[col].isna().sum())
            for col in df.columns
            if df[col].isna().sum() > 0
        }

        skewed = {
            col: float(df[col].skew())
            for col in num_cols
            if abs(df[col].skew()) > 0.5
        }

        # Correlation pairs above 0.9
        corr_matrix     = df[num_cols].corr().abs()
        upper           = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        high_corr_pairs = [
            (c1, c2, round(upper.loc[c2, c1], 4))
            for c1 in upper.columns
            for c2 in upper.index
            if not pd.isna(upper.loc[c2, c1])
            and upper.loc[c2, c1] >= 0.9
        ]

        return {
            "missing_columns": missing,
            "skewed_features": skewed,
            "high_corr_pairs": high_corr_pairs,
            "n_rows":          len(df),
            "n_cols":          len(df.columns),
            "numeric_cols":    num_cols,
        }

    # ─────────────────────────────────────────────────────────────────────
    def _log(self, msg: str):
        if self.config.verbose:
            print(msg)

    @staticmethod
    def _correction_label(iteration: int) -> str:
        labels = {
            0: "None (first run)",
            1: "Outlier IQR capping",
            2: "Power/Log transforms",
            3: "Interaction features",
            4: "Low-variance drop",
        }
        return labels.get(iteration, "Aggressive RobustScaler")
