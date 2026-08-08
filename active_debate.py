# active_debate.py

import json
import requests
from typing import Optional

from ollama_keepalive import KEEP_ALIVE


class TwoAgentDebate:
    """
    Runs a structured debate between two LLM agents (Phi-4 and Mistral)
    over a SHAP diagnostic report.

    Each agent independently analyzes the report and proposes
    a correction strategy. Their arguments are then passed to
    ArbitratorLLM for the final decision.
    """

    # ──────────────────────────────────────────────────────────
    # Debate System Prompt
    # ──────────────────────────────────────────────────────────

    _SYSTEM_PROMPT = """
You are a data science expert analyzing a SHAP diagnostic report for a machine learning pipeline.

Your task is to propose a precise correction strategy based ONLY on the evidence in the report.

You MUST respond in valid JSON with this exact structure:
{
  "drop_columns"       : ["col_a", "col_b"],
  "rescale_columns"    : {"col_c": "RobustScaler", "col_d": "MinMaxScaler"},
  "clip_columns"       : {"col_e": [0, 99]},
  "log_transform_columns" : ["col_f"],
  "next_goal"          : "Brief plain-text description of what to fix next",
  "reasoning"          : "Your justification referencing specific report findings"
}

Rules:
- Use empty list [] or empty dict {} if no action is needed for a field.
- Only reference features that appear in the report.
- Do NOT invent features not mentioned in the diagnostic report.
- RobustScaler is preferred for outlier-heavy features.
- MinMaxScaler is preferred for bounded, clean features.
- Log transform only for right-skewed distributions.
- Clipping must use [lower_percentile, upper_percentile] format.
""".strip()

    # ──────────────────────────────────────────────────────────
    # Init
    # ──────────────────────────────────────────────────────────

    def __init__(
        self,
        ollama_host:    str = "http://localhost:11434",
        agent_a_model:  str = "phi4",
        agent_b_model:  str = "mistral",
        timeout:        int = 120
    ):
        """
        Parameters
        ----------
        ollama_host    : Base URL of the running Ollama instance
        agent_a_model  : First debate agent model name (default: Phi-4)
        agent_b_model  : Second debate agent model name (default: Mistral)
        timeout        : HTTP request timeout in seconds
        """
        self.ollama_host   = ollama_host.rstrip("/")
        self.agent_a_model = agent_a_model
        self.agent_b_model = agent_b_model
        self.timeout       = timeout

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def run(self, shap_report: str) -> dict:
        """
        Runs both agents against the SHAP report and returns
        their arguments as a dict ready for the arbitrator.

        Parameters
        ----------
        shap_report : Plain-text SHAP diagnostic report
                      produced by SHAPPromptBuilder.build()

        Returns
        -------
        dict with keys:
            "agent_a_model"    : str   — model name of agent A
            "agent_b_model"    : str   — model name of agent B
            "agent_a_argument" : dict  — parsed JSON proposal from agent A
            "agent_b_argument" : dict  — parsed JSON proposal from agent B
            "agent_a_raw"      : str   — raw text from agent A (for logging)
            "agent_b_raw"      : str   — raw text from agent B (for logging)
            "agent_a_failed"   : bool  — True if agent A returned invalid JSON
            "agent_b_failed"   : bool  — True if agent B returned invalid JSON
        """
        print(f"[TwoAgentDebate] Agent A ({self.agent_a_model}) is analyzing...")
        raw_a, parsed_a, failed_a = self._query_agent(
            model  = self.agent_a_model,
            report = shap_report
        )

        print(f"[TwoAgentDebate] Agent B ({self.agent_b_model}) is analyzing...")
        raw_b, parsed_b, failed_b = self._query_agent(
            model  = self.agent_b_model,
            report = shap_report
        )

        return {
            "agent_a_model"    : self.agent_a_model,
            "agent_b_model"    : self.agent_b_model,
            "agent_a_argument" : parsed_a,
            "agent_b_argument" : parsed_b,
            "agent_a_raw"      : raw_a,
            "agent_b_raw"      : raw_b,
            "agent_a_failed"   : failed_a,
            "agent_b_failed"   : failed_b,
        }

    # ──────────────────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────────────────

    def _query_agent(
        self,
        model:  str,
        report: str
    ) -> tuple:
        """
        Sends the SHAP report to a single LLM agent via Ollama
        and returns the raw response, parsed JSON dict, and
        a failure flag.

        Returns
        -------
        (raw_text: str, parsed: dict, failed: bool)
        """
        payload = {
            "model"  : model,
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
                    "content": (
                        "Here is the SHAP Diagnostic Report.\n"
                        "Propose your correction strategy:\n\n"
                        f"{report}"
                    )
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
            parsed, failed = self._parse_json_response(raw_text, model)
            return raw_text, parsed, failed

        except requests.exceptions.Timeout:
            print(f"[TwoAgentDebate] WARNING: {model} timed out.")
            return "", self._empty_proposal(), True

        except requests.exceptions.RequestException as e:
            print(f"[TwoAgentDebate] WARNING: {model} request failed — {e}")
            return "", self._empty_proposal(), True

    def _parse_json_response(
        self,
        raw_text: str,
        model:    str
    ) -> tuple:
        """
        Attempts to extract and parse JSON from the LLM response.
        Handles cases where the model wraps JSON in markdown code blocks.

        Returns
        -------
        (parsed_dict: dict, failed: bool)
        """
        # Strip markdown code fences if present
        clean = raw_text
        if "```json" in clean:
            clean = clean.split("```json")[-1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[-2].strip() if clean.count("```") >= 2 else clean

        try:
            parsed = json.loads(clean)
            # A list, string or number is valid JSON and not a proposal.
            # `_fill_defaults` subscripts it and raises an uncaught TypeError,
            # which kills the debate instead of degrading to an empty proposal.
            # It also escapes upstream: ArbitratorLLM.decide() calls
            # _fill_defaults on a surviving agent's argument, so a non-dict
            # here would crash the arbitration too.
            if not isinstance(parsed, dict):
                print(
                    f"[TwoAgentDebate] WARNING: {model} returned valid JSON of "
                    f"type {type(parsed).__name__}, not an object. "
                    f"Using empty proposal."
                )
                return self._empty_proposal(), True
            # Ensure all required keys exist with safe defaults
            parsed = self._fill_defaults(parsed)
            print(f"[TwoAgentDebate] {model} responded successfully.")
            return parsed, False

        except json.JSONDecodeError:
            print(
                f"[TwoAgentDebate] WARNING: {model} returned invalid JSON. "
                f"Using empty proposal."
            )
            return self._empty_proposal(), True

    def _fill_defaults(self, proposal: dict) -> dict:
        """
        Ensures all required keys are present in the proposal dict.
        Prevents KeyError downstream in ArbitratorLLM.
        """
        defaults = self._empty_proposal()
        for key, default_val in defaults.items():
            if key not in proposal:
                proposal[key] = default_val
        return proposal

    def _empty_proposal(self) -> dict:
        """
        Returns a safe empty correction proposal.
        Used when an agent fails or times out.
        """
        return {
            "drop_columns"          : [],
            "rescale_columns"       : {},
            "clip_columns"          : {},
            "log_transform_columns" : [],
            "next_goal"             : "No recommendation — agent failed.",
            "reasoning"             : "Agent did not return a valid response."
        }
