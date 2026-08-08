# arbitrator.py

import json
import requests
from typing import Optional

from ollama_keepalive import KEEP_ALIVE


class ArbitratorLLM:
    """
    Receives both agent proposals from TwoAgentDebate and
    the original SHAP diagnostic report, then produces a
    single authoritative correction strategy as the final decision.

    Uses a third LLM (default: Phi-4) acting as a neutral judge.
    """

    # ──────────────────────────────────────────────────────────
    # Arbitrator System Prompt
    # ──────────────────────────────────────────────────────────

    _SYSTEM_PROMPT = """
You are a senior data science arbitrator. You will receive:
  1. A SHAP Diagnostic Report describing the current state of an ML pipeline.
  2. Two correction proposals from two independent expert agents (Agent A and Agent B).

Your job is to evaluate both proposals critically and produce ONE final correction strategy.

Rules for arbitration:
- Prefer the proposal with stronger evidence from the SHAP report.
- If both agents agree on an action, include it in the final decision.
- If agents disagree, choose the safer, more conservative option unless one has clear evidence.
- NEVER invent features or actions not grounded in the SHAP report.
- RobustScaler is preferred for outlier-heavy features.
- Clipping is safer than dropping when evidence is ambiguous.
- Do NOT drop a feature unless at least one agent proposes it WITH clear reasoning.

You MUST respond in valid JSON with this exact structure:
{
  "drop_columns"          : ["col_a"],
  "rescale_columns"       : {"col_b": "RobustScaler"},
  "clip_columns"          : {"col_c": [1, 99]},
  "log_transform_columns" : ["col_d"],
  "next_goal"             : "Brief plain-text description of what to fix next",
  "reasoning"             : "Your full arbitration reasoning referencing both agents and the SHAP report",
  "consensus"             : true
}

Set "consensus" to true if both agents substantially agreed, false if you had to break a tie.
Use empty list [] or empty dict {} where no action is needed.
""".strip()

    # ──────────────────────────────────────────────────────────
    # Init
    # ──────────────────────────────────────────────────────────

    def __init__(
        self,
        ollama_host:       str = "http://localhost:11434",
        arbitrator_model:  str = "phi4",
        timeout:           int = 120
    ):
        """
        Parameters
        ----------
        ollama_host       : Base URL of the running Ollama instance
        arbitrator_model  : LLM model to use as arbitrator (default: Phi-4)
        timeout           : HTTP request timeout in seconds
        """
        self.ollama_host      = ollama_host.rstrip("/")
        self.arbitrator_model = arbitrator_model
        self.timeout          = timeout

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def decide(
        self,
        shap_report:     str,
        debate_result:   dict
    ) -> dict:
        """
        Takes the SHAP report and the full debate result from
        TwoAgentDebate.run() and returns a final arbitrated
        correction strategy.

        Parameters
        ----------
        shap_report   : Plain-text SHAP diagnostic report from SHAPPromptBuilder
        debate_result : Dict returned by TwoAgentDebate.run()

        Returns
        -------
        dict with keys:
            "final_decision"    : dict  — final correction strategy
            "consensus"         : bool  — did agents agree?
            "arbitrator_model"  : str   — model used to arbitrate
            "arbitrator_raw"    : str   — raw LLM response (for logging)
            "arbitrator_failed" : bool  — True if arbitrator returned invalid JSON
            "fallback_used"     : bool  — True if fallback logic was triggered
        """

        # ── Check if both agents failed ─────────────────────────
        agent_a_failed = debate_result.get("agent_a_failed", True)
        agent_b_failed = debate_result.get("agent_b_failed", True)

        if agent_a_failed and agent_b_failed:
            print(
                "[ArbitratorLLM] Both agents failed. "
                "Returning empty decision — no corrections applied."
            )
            return self._build_result(
                decision         = self._empty_decision(),
                consensus        = False,
                raw              = "",
                arbitrator_failed= False,
                fallback_used    = True
            )

        # ── If one agent failed, use the surviving agent directly ─
        if agent_a_failed or agent_b_failed:
            surviving_key = "agent_b_argument" if agent_a_failed else "agent_a_argument"
            surviving     = debate_result.get(surviving_key, self._empty_decision())
            surviving_model = (
                debate_result.get("agent_b_model") if agent_a_failed
                else debate_result.get("agent_a_model")
            )
            print(
                f"[ArbitratorLLM] One agent failed. "
                f"Using surviving agent ({surviving_model}) directly."
            )
            surviving = self._fill_defaults(surviving)
            return self._build_result(
                decision         = surviving,
                consensus        = False,
                raw              = "",
                arbitrator_failed= False,
                fallback_used    = True
            )

        # ── Both agents responded — run full arbitration ─────────
        user_prompt = self._build_user_prompt(shap_report, debate_result)

        print(
            f"[ArbitratorLLM] Arbitrating with "
            f"{self.arbitrator_model}..."
        )

        raw, parsed, failed = self._query_arbitrator(user_prompt)

        if failed:
            # Arbitrator itself failed — fallback to conservative merge
            print(
                "[ArbitratorLLM] Arbitrator returned invalid JSON. "
                "Falling back to conservative merge."
            )
            merged = self._conservative_merge(
                debate_result.get("agent_a_argument", {}),
                debate_result.get("agent_b_argument", {})
            )
            return self._build_result(
                decision         = merged,
                consensus        = False,
                raw              = raw,
                arbitrator_failed= True,
                fallback_used    = True
            )

        return self._build_result(
            decision         = parsed,
            consensus        = parsed.get("consensus", False),
            raw              = raw,
            arbitrator_failed= False,
            fallback_used    = False
        )

    # ──────────────────────────────────────────────────────────
    # Private Helpers — LLM Query
    # ──────────────────────────────────────────────────────────

    def _query_arbitrator(self, user_prompt: str) -> tuple:
        """
        Sends the arbitration prompt to the LLM and returns
        the raw response, parsed JSON, and a failure flag.

        Returns
        -------
        (raw_text: str, parsed: dict, failed: bool)
        """
        payload = {
            "model"  : self.arbitrator_model,
            "stream" : False,
            # Free the VRAM as soon as the reply is back — see ollama_keepalive.
            "keep_alive": KEEP_ALIVE,
            "messages": [
                {
                    "role"   : "system",
                    "content": self._SYSTEM_PROMPT
                },
                {
                    "role"   : "user",
                    "content": user_prompt
                }
            ]
        }

        try:
            response = requests.post(
                url     = f"{self.ollama_host}/api/chat",
                json    = payload,
                timeout = self.timeout
            )
            response.raise_for_status()
            raw_text = response.json()["message"]["content"].strip()
            parsed, failed = self._parse_json_response(raw_text)
            return raw_text, parsed, failed

        except requests.exceptions.Timeout:
            print(f"[ArbitratorLLM] WARNING: {self.arbitrator_model} timed out.")
            return "", self._empty_decision(), True

        except requests.exceptions.RequestException as e:
            print(f"[ArbitratorLLM] WARNING: Request failed — {e}")
            return "", self._empty_decision(), True

    # ──────────────────────────────────────────────────────────
    # Private Helpers — Prompt Building
    # ──────────────────────────────────────────────────────────

    def _build_user_prompt(
        self,
        shap_report:   str,
        debate_result: dict
    ) -> str:
        """
        Constructs the full user prompt sent to the arbitrator,
        including the SHAP report and both agent proposals.
        """
        agent_a_model = debate_result.get("agent_a_model", "Agent A")
        agent_b_model = debate_result.get("agent_b_model", "Agent B")
        agent_a_arg   = json.dumps(
            debate_result.get("agent_a_argument", {}), indent=2
        )
        agent_b_arg   = json.dumps(
            debate_result.get("agent_b_argument", {}), indent=2
        )

        return (
            f"=== SHAP Diagnostic Report ===\n"
            f"{shap_report}\n\n"
            f"=== Agent A Proposal ({agent_a_model}) ===\n"
            f"{agent_a_arg}\n\n"
            f"=== Agent B Proposal ({agent_b_model}) ===\n"
            f"{agent_b_arg}\n\n"
            f"Now produce your final arbitrated correction strategy."
        )

    # ──────────────────────────────────────────────────────────
    # Private Helpers — JSON Parsing
    # ──────────────────────────────────────────────────────────

    def _parse_json_response(self, raw_text: str) -> tuple:
        """
        Attempts to extract and parse JSON from the LLM response.
        Strips markdown code fences if present.

        Returns
        -------
        (parsed_dict: dict, failed: bool)
        """
        clean = raw_text
        if "```json" in clean:
            clean = clean.split("```json")[-1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[-2].strip() if clean.count("```") >= 2 else clean

        try:
            parsed = json.loads(clean)
            parsed = self._fill_defaults(parsed)
            print(
                f"[ArbitratorLLM] {self.arbitrator_model} "
                f"arbitrated successfully."
            )
            return parsed, False

        except json.JSONDecodeError:
            print(
                "[ArbitratorLLM] WARNING: Arbitrator returned invalid JSON. "
                "Triggering fallback."
            )
            return self._empty_decision(), True

    # ──────────────────────────────────────────────────────────
    # Private Helpers — Fallback & Defaults
    # ──────────────────────────────────────────────────────────

    def _conservative_merge(
        self,
        proposal_a: dict,
        proposal_b: dict
    ) -> dict:
        """
        Fallback strategy when the arbitrator itself fails.
        Merges two proposals conservatively:
          - Only drops columns both agents agree to drop.
          - Merges rescale, clip, and log_transform from both agents.
          - Prefers RobustScaler if agents disagree on scaler type.
        """
        # Drop only if BOTH agents agree
        drop_a = set(proposal_a.get("drop_columns", []))
        drop_b = set(proposal_b.get("drop_columns", []))
        agreed_drops = list(drop_a & drop_b)

        # Merge rescale — prefer RobustScaler on conflict
        rescale = {}
        for col, scaler in proposal_a.get("rescale_columns", {}).items():
            rescale[col] = scaler
        for col, scaler in proposal_b.get("rescale_columns", {}).items():
            if col in rescale and rescale[col] != scaler:
                rescale[col] = "RobustScaler"
            else:
                rescale[col] = scaler

        # Merge clip — prefer wider bounds on conflict
        clip = {}
        for col, bounds in proposal_a.get("clip_columns", {}).items():
            clip[col] = bounds
        for col, bounds in proposal_b.get("clip_columns", {}).items():
            if col in clip:
                # Take the wider range
                clip[col] = [
                    min(clip[col][0], bounds[0]),
                    max(clip[col][1], bounds[1])
                ]
            else:
                clip[col] = bounds

        # Merge log transform — union of both
        log_a = set(proposal_a.get("log_transform_columns", []))
        log_b = set(proposal_b.get("log_transform_columns", []))
        log_merged = list(log_a | log_b)

        merged = {
            "drop_columns"          : agreed_drops,
            "rescale_columns"       : rescale,
            "clip_columns"          : clip,
            "log_transform_columns" : log_merged,
            "next_goal"             : "Conservative merge — arbitrator fallback applied.",
            "reasoning"             : (
                "Arbitrator LLM failed. Applied conservative merge: "
                "only dropped columns both agents agreed on, "
                "merged rescale/clip/log transforms with safe defaults."
            ),
            "consensus"             : False
        }
        return merged

    def _fill_defaults(self, decision: dict) -> dict:
        """
        Ensures all required keys are present in the decision dict.
        Prevents KeyError downstream in feedback_controller.py.
        """
        defaults = self._empty_decision()
        for key, val in defaults.items():
            if key not in decision:
                decision[key] = val
        return decision

    def _empty_decision(self) -> dict:
        """
        Returns a safe empty decision dict.
        Used when all agents and the arbitrator fail.
        """
        return {
            "drop_columns"          : [],
            "rescale_columns"       : {},
            "clip_columns"          : {},
            "log_transform_columns" : [],
            "next_goal"             : "No corrections — all agents failed.",
            "reasoning"             : "No valid proposals were received.",
            "consensus"             : False
        }

    def _build_result(
        self,
        decision:          dict,
        consensus:         bool,
        raw:               str,
        arbitrator_failed: bool,
        fallback_used:     bool
    ) -> dict:
        """
        Wraps the final decision into the standard return structure.
        """
        return {
            "final_decision"    : decision,
            "consensus"         : consensus,
            "arbitrator_model"  : self.arbitrator_model,
            "arbitrator_raw"    : raw,
            "arbitrator_failed" : arbitrator_failed,
            "fallback_used"     : fallback_used
        }
