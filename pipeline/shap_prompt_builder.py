# pipeline/shap_prompt_builder.py

from typing import Optional


class SHAPPromptBuilder:
    """
    Converts SHAP audit results and iteration history
    into a structured diagnostic report for LLM consumption.
    Reads output produced by shap_feedback.py.
    """

    def build(
        self,
        iteration: int,
        current_score: float,
        previous_score: Optional[float],
        shap_top5: list,
        ghost_features: list,
        damaged_features: list,
        outlier_features: list,
        history: list
    ) -> str:
        """
        Builds a structured plain-text diagnostic report
        to be injected into LLM system/user prompts.

        Parameters
        ----------
        iteration        : Current feedback loop iteration number
        current_score    : Model score for this iteration
        previous_score   : Model score from the previous iteration (or None)
        shap_top5        : Top 5 most impactful features by SHAP value
        ghost_features   : Features with high SHAP but high inter-correlation
        damaged_features : Features whose SHAP contribution dropped 80%+ vs prev iter
        outlier_features : Features where outliers dominate SHAP contribution
        history          : List of previous iteration summary dicts

        Returns
        -------
        str : Formatted diagnostic report string
        """

        score_delta_text = self._format_score_delta(current_score, previous_score)
        ghost_text       = self._format_feature_issue(
            label       = "Ghost Features",
            features    = ghost_features,
            description = (
                "High SHAP value but highly correlated with other features. "
                "Likely redundant — candidate for removal."
            )
        )
        damaged_text     = self._format_feature_issue(
            label       = "Damaged Features",
            features    = damaged_features,
            description = (
                "SHAP contribution dropped 80%+ compared to previous iteration. "
                "A recent transformation may have destroyed the signal."
            )
        )
        outlier_text     = self._format_feature_issue(
            label       = "Outlier-Distorted Features",
            features    = outlier_features,
            description = (
                "A small number of data points dominate the SHAP contribution. "
                "Model may be overfitting to outliers."
            )
        )
        history_text     = self._format_history(history)

        report = f"""
=== SHAP Diagnostic Report — Iteration {iteration} ===

Model Score : {current_score:.4f} {score_delta_text}
SHAP Top 5  : {', '.join(shap_top5) if shap_top5 else 'None'}

--- Detected Issues ---
{ghost_text}
{damaged_text}
{outlier_text}

--- Previous Iteration History (last 3) ---
{history_text}

=== End of Report ===
""".strip()

        return report

    # ──────────────────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────────────────

    def _format_score_delta(
        self,
        current_score: float,
        previous_score: Optional[float]
    ) -> str:
        """
        Returns a human-readable score change indicator.
        Example: '(prev: 0.8120 → IMPROVED +0.0150)'
        """
        if previous_score is None:
            return "(first iteration)"

        delta     = current_score - previous_score
        direction = "IMPROVED" if delta > 0 else "DEGRADED"
        return f"(prev: {previous_score:.4f} → {direction} {delta:+.4f})"

    def _format_feature_issue(
        self,
        label: str,
        features: list,
        description: str
    ) -> str:
        """
        Returns a formatted block for a category of SHAP issues.
        Shows 'None detected' if the feature list is empty.
        """
        if not features:
            return f"[OK] {label}: None detected."

        feature_list = ', '.join(features)
        return (
            f"[WARN] {label}: {feature_list}\n"
            f"       Reason: {description}"
        )

    def _format_history(self, history: list) -> str:
        """
        Returns a formatted summary of the last 3 iterations.
        """
        if not history:
            return "  No history available."

        lines = []
        for entry in history[-3:]:
            iter_num    = entry.get("iteration",   "?")
            score       = entry.get("score",       None)
            corrections = entry.get("corrections", "none")
            score_str   = f"{score:.4f}" if isinstance(score, float) else str(score)
            lines.append(
                f"  Iter {iter_num}: "
                f"score={score_str}, "
                f"corrections={corrections}"
            )

        return "\n".join(lines)
