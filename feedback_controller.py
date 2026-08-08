# feedback_controller.py

from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np


@dataclass
class FeedbackConfig:
    task_type:         Literal["regression", "classification"] = "regression"
    metric_threshold:  float = 0.95
    max_iterations:    int   = 5
    models:            list  = field(default_factory=lambda: ["xgboost", "neural_network"])
    verbose:           bool  = True

    # ── Core Settings ───────────────────────────────────────────
    target_col:        str   = "target"
    ollama_host:       str   = "http://localhost:11434"
    use_local_llm:     bool  = False

    # ── Debate & Arbitrator Settings ────────────────────────────
    debate_agent_a:    str   = "phi4"
    debate_agent_b:    str   = "mistral"
    arbitrator_model:  str   = "phi4"
    llm_timeout:       int   = 120


class AutoEDAFeedbackController:
    """
    Closed-loop EDA feedback system.

    When use_local_llm = False:
        Uses FeedbackAnalyzer (rule-based) for corrections.

    When use_local_llm = True:
        Runs the full LLM debate pipeline:
            SHAPPromptBuilder → TwoAgentDebate → ArbitratorLLM
        The arbitrated decision is passed to FeedbackAnalyzer
        to apply corrections to the DataFrame.
    """

    def __init__(self, config: FeedbackConfig):
        self.config         = config
        self.iteration      = 0
        self.history        = []
        self.best_score     = -np.inf
        self.best_eda_state = None

        self._prompt_builder = None
        self._debate         = None
        self._arbitrator     = None

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def run(self, raw_df):
        """
        Main feedback control loop.

        Parameters
        ----------
        raw_df : pd.DataFrame — raw input data

        Returns
        -------
        (df, model_name, history)
        """
        from eda_pipeline      import FeedbackEDAPipeline
        from model_trainer     import ModelTrainer
        from feedback_analyzer import FeedbackAnalyzer

        df       = raw_df.copy()
        eda      = FeedbackEDAPipeline(self.config)   # ✅ config passed
        trainer  = ModelTrainer(self.config)           # ✅ config passed
        analyzer = FeedbackAnalyzer(self.config)

        if self.config.use_local_llm:
            self._init_llm_components()

        while self.iteration < self.config.max_iterations:
            self.iteration += 1

            if self.config.verbose:
                print(f"\n{'='*52}")
                print(f"🔁 Feedback Iteration #{self.iteration}")
                print(f"{'='*52}")

            # ── Step 1: Run EDA ──────────────────────────────────
            # Returns: (eda_report, model, model_name, score, feature_imp)
            eda_report, eda_model, eda_model_name, eda_score, feature_imp = eda.run(df)

            # ── Step 2: Train & Evaluate Models ─────────────────
            score, model_name = trainer.train_and_evaluate(df)

            # ── Step 3: Log History ──────────────────────────────
            previous_score = (
                self.history[-1]["score"] if self.history else None
            )
            self.history.append({
                "iteration"   : self.iteration,
                "model"       : model_name,
                "score"       : score,
                "eda_report"  : eda_report,
                "corrections" : "none"
            })

            if self.config.verbose:
                metric = "R²" if self.config.task_type == "regression" else "Accuracy"
                print(f"📊 {metric}: {score:.4f} | Model: {model_name}")

            # ── Step 4: Check Threshold ──────────────────────────
            if score >= self.config.metric_threshold:
                print(
                    f"\n✅ Threshold met at iteration "
                    f"{self.iteration}! Score: {score:.4f}"
                )
                # ✅ FIX 4: return df  (not undefined processed_df)
                return df, model_name, self.history

            # ── Step 5: Track Best ───────────────────────────────
            if score > self.best_score:
                self.best_score     = score
                self.best_eda_state = eda_report

            # ── Step 6: Feedback Corrections ────────────────────
            if self.config.use_local_llm:
                df = self._run_llm_feedback(
                    df             = df,
                    eda_report     = eda_report,
                    score          = score,
                    previous_score = previous_score,
                    analyzer       = analyzer
                )
            else:
                df = analyzer.suggest_corrections(
                    df, eda_report, score, self.iteration
                )

        print(f"\n⚠️ Max iterations reached. Best Score: {self.best_score:.4f}")
        return df, None, self.history

    # ──────────────────────────────────────────────────────────
    # LLM Feedback Pipeline
    # ──────────────────────────────────────────────────────────

    def _run_llm_feedback(
        self,
        df,
        eda_report:     dict,
        score:          float,
        previous_score: Optional[float],
        analyzer
    ):
        """
        Runs the full SHAP → Debate → Arbitrate → Correct pipeline.
        """

        # ── 1. Build SHAP Diagnostic Report ─────────────────────
        shap_report = self._prompt_builder.build(
            iteration        = self.iteration,
            current_score    = score,
            previous_score   = previous_score,
            shap_top5        = eda_report.get("shap_top5",        []),
            ghost_features   = eda_report.get("ghost_features",   []),
            damaged_features = eda_report.get("damaged_features", []),
            outlier_features = eda_report.get("outlier_features", []),
            history          = self.history
        )

        if self.config.verbose:
            print("\n📋 SHAP Diagnostic Report Generated.")
            print(shap_report)

        # ── 2. Run Two-Agent Debate ──────────────────────────────
        debate_result = self._debate.run(shap_report)

        if self.config.verbose:
            self._log_debate_result(debate_result)

        # ── 3. Arbitrate Final Decision ──────────────────────────
        arbitration = self._arbitrator.decide(
            shap_report   = shap_report,
            debate_result = debate_result
        )

        final_decision = arbitration["final_decision"]

        if self.config.verbose:
            self._log_arbitration_result(arbitration)

        # ── 4. Update history with corrections summary ───────────
        self._update_history_corrections(final_decision)

        # ── 5. Apply Corrections via FeedbackAnalyzer ───────────
        df = analyzer.apply_llm_corrections(df, final_decision)

        return df

    # ──────────────────────────────────────────────────────────
    # LLM Component Init
    # ──────────────────────────────────────────────────────────

    def _init_llm_components(self):
        from pipeline.shap_prompt_builder import SHAPPromptBuilder
        from active_debate                import TwoAgentDebate
        from arbitrator                   import ArbitratorLLM

        self._prompt_builder = SHAPPromptBuilder()

        self._debate = TwoAgentDebate(
            ollama_host   = self.config.ollama_host,
            agent_a_model = self.config.debate_agent_a,
            agent_b_model = self.config.debate_agent_b,
            timeout       = self.config.llm_timeout
        )

        self._arbitrator = ArbitratorLLM(
            ollama_host      = self.config.ollama_host,
            arbitrator_model = self.config.arbitrator_model,
            timeout          = self.config.llm_timeout
        )

        if self.config.verbose:
            print(
                f"\n🤖 LLM Pipeline Initialized:\n"
                f"   Agent A     : {self.config.debate_agent_a}\n"
                f"   Agent B     : {self.config.debate_agent_b}\n"
                f"   Arbitrator  : {self.config.arbitrator_model}\n"
                f"   Ollama Host : {self.config.ollama_host}"
            )

    # ──────────────────────────────────────────────────────────
    # Logging Helpers
    # ──────────────────────────────────────────────────────────

    def _log_debate_result(self, debate_result: dict):
        a_model  = debate_result.get("agent_a_model",    "Agent A")
        b_model  = debate_result.get("agent_b_model",    "Agent B")
        a_failed = debate_result.get("agent_a_failed",   False)
        b_failed = debate_result.get("agent_b_failed",   False)

        print(f"\n🗣️  Debate Results:")
        print(
            f"   {a_model}: "
            f"{'❌ FAILED' if a_failed else '✅ OK'} | "
            f"goal → {debate_result.get('agent_a_argument', {}).get('next_goal', 'N/A')}"
        )
        print(
            f"   {b_model}: "
            f"{'❌ FAILED' if b_failed else '✅ OK'} | "
            f"goal → {debate_result.get('agent_b_argument', {}).get('next_goal', 'N/A')}"
        )

    def _log_arbitration_result(self, arbitration: dict):
        decision  = arbitration.get("final_decision",    {})
        consensus = arbitration.get("consensus",         False)
        fallback  = arbitration.get("fallback_used",     False)
        model     = arbitration.get("arbitrator_model",  "?")

        print(f"\n⚖️  Arbitration Result ({model}):")
        print(f"   Consensus   : {'✅ Yes' if consensus else '❌ No'}")
        print(f"   Fallback    : {'⚠️ Yes' if fallback  else '✅ No'}")
        print(f"   Drop        : {decision.get('drop_columns',          [])}")
        print(f"   Rescale     : {decision.get('rescale_columns',       {})}")
        print(f"   Clip        : {decision.get('clip_columns',          {})}")
        print(f"   Log Trans.  : {decision.get('log_transform_columns', [])}")
        print(f"   Next Goal   : {decision.get('next_goal', 'N/A')}")

    def _update_history_corrections(self, final_decision: dict):
        if not self.history:
            return

        parts = []
        if final_decision.get("drop_columns"):
            parts.append(f"drop_columns={final_decision['drop_columns']}")
        if final_decision.get("rescale_columns"):
            parts.append(f"rescale_columns={final_decision['rescale_columns']}")
        if final_decision.get("clip_columns"):
            parts.append(f"clip_columns={final_decision['clip_columns']}")
        if final_decision.get("log_transform_columns"):
            parts.append(f"log_transform={final_decision['log_transform_columns']}")

        self.history[-1]["corrections"] = (
            ", ".join(parts) if parts else "none"
        )
