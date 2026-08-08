# pipeline/local_llm_engine.py

import json
import re
import time
import logging
import numpy as np
import pandas as pd
import concurrent.futures
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field
from tenacity import retry, stop_after_attempt, wait_exponential

from ollama_keepalive import KEEP_ALIVE

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — ENUMS
# ─────────────────────────────────────────────────────────────────────────────
class ModelRole(Enum):
    PHI4    = "phi4"
    MISTRAL = "mistral"
    GEMMA2  = "gemma2"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — RTX4090 CONFIG
# ─────────────────────────────────────────────────────────────────────────────
RTX4090_CONFIG = {
    ModelRole.PHI4: {
        "model_tag":     "phi4:latest",
        "num_gpu":        999,
        "num_thread":     8,
        "num_ctx":        8192,
        "temperature":    0.05,
        "num_predict":    2048,
        "top_p":          0.9,
        "repeat_penalty": 1.1,
        "f16_kv":         True,
        "use_mmap":       True,
    },
    ModelRole.MISTRAL: {
        "model_tag":     "mistral:latest",
        "num_gpu":        999,
        "num_thread":     8,
        "num_ctx":        4096,
        "temperature":    0.10,
        "num_predict":    2048,
        "top_p":          0.9,
        "repeat_penalty": 1.1,
        "f16_kv":         True,
        "use_mmap":       True,
    },
    ModelRole.GEMMA2: {
        "model_tag":     "gemma2:latest",
        "num_gpu":        999,
        "num_thread":     8,
        "num_ctx":        4096,
        "temperature":    0.05,    # very low = deterministic tiebreaker
        "num_predict":    1024,
        "top_p":          0.9,
        "repeat_penalty": 1.1,
        "f16_kv":         True,
        "use_mmap":       True,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPTS = {
    ModelRole.PHI4: """You are an expert data scientist with deep knowledge of
preprocessing pipelines. Your role is ANALYTICAL REASONING: analyze the
dataset profile carefully, justify each decision, and return a valid JSON object.
Focus on correctness and detailed reasoning. Always output valid JSON only.""",

    ModelRole.MISTRAL: """You are a pragmatic data scientist. Your role is PATTERN
MATCHING and VALIDATION: quickly assess the dataset characteristics, apply
proven heuristics, and return a valid JSON object.
Be concise and practical. Always output valid JSON only.""",

    ModelRole.GEMMA2: """You are a precise data scientist acting as a tiebreaker.
Analyze the dataset profile and return the single best preprocessing decision
as valid JSON only. Be definitive and concise.""",
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────
DECISION_PROMPT_TEMPLATE = """
Analyze this dataset profile and return preprocessing decisions as JSON.

## Dataset Profile:
{profile_json}

## Task: {task}

## REQUIRED JSON OUTPUT (return ONLY this JSON, no other text):
{{
  "imputation_strategy": "mean OR median OR knn OR missforest",
  "imputation_reasoning": "brief explanation",
  "knn_neighbors": 5,
  "power_transform": "yeo-johnson OR box-cox OR none",
  "outlier_method": "iqr OR zscore OR isolation_forest OR none",
  "outlier_threshold": 1.5,
  "correlation_threshold": 0.90,
  "feature_engineering": {{
    "polynomial": false,
    "poly_degree": 2,
    "interactions_only": false,
    "interaction_features": false
  }},
  "encoding": {{
    "onehot_features": [],
    "ordinal_features": []
  }},
  "scaler": "standard OR minmax OR robust",
  "reasoning_summary": "2-3 sentence summary",
  "confidence": 0.85
}}

Rules:
- Use yeo-johnson (not box-cox) if any feature has negative values
- Use missforest if missing > 20% OR if data is MNAR
- Use robust scaler if many outliers detected
- Add polynomial features only if n_rows > 100 and strong nonlinearity suspected
"""

TIEBREAK_PROMPT_TEMPLATE = """
You are a tiebreaker between two AI models on preprocessing decisions.

## Dataset Profile:
{profile_json}

## Conflicting Decisions:
{conflict_lines}

## Your Task:
For EACH conflicting key, choose the BEST value based on the dataset profile.
Return ONLY a JSON object with the conflicting keys and your chosen values.

Example output format:
{{
  "imputation_strategy": "knn",
  "scaler": "robust"
}}

Return ONLY the JSON. No explanation. No extra text.
"""


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LLMDecision:
    """Holds a single model's preprocessing decision."""
    model:                  str
    imputation_strategy:    str   = "median"
    imputation_reasoning:   str   = ""
    knn_neighbors:          int   = 5
    power_transform:        str   = "yeo-johnson"
    outlier_method:         str   = "iqr"
    outlier_threshold:      float = 1.5
    correlation_threshold:  float = 0.90
    feature_engineering:    dict  = field(default_factory=lambda: {
        "polynomial":           False,
        "poly_degree":          2,
        "interactions_only":    False,
        "interaction_features": False,
    })
    encoding: dict = field(default_factory=lambda: {
        "onehot_features":  [],
        "ordinal_features": [],
    })
    scaler:            str   = "standard"
    reasoning_summary: str   = ""
    raw_response:      str   = ""
    confidence:        float = 0.0
    latency_ms:        float = 0.0


@dataclass
class EnsembleDecision:
    """Final merged decision from Phi-4 + Mistral + optional Gemma2 tiebreak."""
    final:            LLMDecision
    phi4_decision:    LLMDecision
    mistral_decision: LLMDecision
    conflicts:        list  = field(default_factory=list)
    agreement_score:  float = 0.0
    tiebreak_used:    bool  = False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — OLLAMA CLIENT
# ─────────────────────────────────────────────────────────────────────────────
class OllamaClient:
    """RTX 4090-optimized Ollama HTTP client."""

    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self._verify_ollama()

    # ── Connection check ─────────────────────────────────────────────────────
    def _verify_ollama(self):
        import httpx
        try:
            resp      = httpx.get(f"{self.host}/api/tags", timeout=5)
            available = [m["name"] for m in resp.json().get("models", [])]
            logger.info(f"[OllamaClient] Connected. Models available: {available}")
        except Exception as e:
            raise ConnectionError(
                f"Ollama not reachable at {self.host}.\n"
                f"Fix: run  `ollama serve`\n"
                f"Error: {e}"
            )

    # ── Generation with retry ─────────────────────────────────────────────────
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate(
        self,
        model_role: ModelRole,
        prompt:     str,
        system:     str = "",
    ) -> tuple[str, float]:
        """
        Send a prompt to Ollama using the model-specific RTX4090 config.
        Returns (response_text, latency_ms).
        """
        import httpx

        cfg = RTX4090_CONFIG[model_role]
        t0  = time.perf_counter()

        # Merge system prompt into user prompt if provided
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        # Build options — exclude non-Ollama keys
        options = {k: v for k, v in cfg.items() if k != "model_tag"}

        payload = {
            "model":      cfg["model_tag"],
            "prompt":     full_prompt,
            "stream":     False,
            "options":    options,
            # Free the VRAM as soon as the reply is back — see ollama_keepalive.
            "keep_alive": KEEP_ALIVE,
        }

        resp = httpx.post(
            f"{self.host}/api/generate",
            json=payload,
            timeout=120
        )
        resp.raise_for_status()

        latency = (time.perf_counter() - t0) * 1000
        raw     = resp.json().get("response", "")

        logger.debug(f"[{model_role.value}] raw response length: {len(raw)} chars")
        return raw, latency

    # ── JSON extractor ────────────────────────────────────────────────────────
    def extract_json(self, text: str) -> dict:
        """Robustly extract a JSON object from messy LLM output."""
        # Strip markdown code fences
        cleaned  = re.sub(r"```(?:json)?", "", text).strip().strip("`")

        start = cleaned.find("{")
        end   = cleaned.rfind("}") + 1

        if start == -1 or end == 0:
            raise ValueError(f"No JSON object found in response:\n{text[:300]}")

        json_str = cleaned[start:end]

        # Fix common LLM JSON quirks
        json_str = re.sub(r",\s*}",    "}",    json_str)   # trailing comma in object
        json_str = re.sub(r",\s*]",    "]",    json_str)   # trailing comma in array
        json_str = re.sub(r"//.*?\n",  "\n",   json_str)   # JS-style comments
        json_str = re.sub(r"\bTrue\b",  "true",  json_str) # Python bool → JSON
        json_str = re.sub(r"\bFalse\b", "false", json_str)
        json_str = re.sub(r"\bNone\b",  "null",  json_str)

        return json.loads(json_str)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — ENSEMBLE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class LocalEnsembleLLMEngine:
    """
    Phi-4 + Mistral ensemble with Gemma2 tiebreaker.

    Flow:
        profile_dataset(df)
            ↓
        run_ensemble(profile)  ← called by orchestrator
            ↓
        analyze_and_decide(profile, task)
            ↓
        [Phi-4 ‖ Mistral] in parallel threads
            ↓
        _resolve_ensemble(phi4, mistral, profile)
            ↓ (only if conflicts exist)
        _tiebreak_with_gemma2(conflicts, profile)
            ↓
        EnsembleDecision
    """

    CONFLICT_KEYS = [
        "imputation_strategy",
        "power_transform",
        "outlier_method",
        "scaler",
    ]

    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.client = OllamaClient(host=ollama_host)

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT — called by orchestrator.py
    # ─────────────────────────────────────────────────────────────────────────
    def run_ensemble(
        self,
        profile: dict,
        task:    str = "classification"
    ) -> EnsembleDecision:
        """
        Main public method.
        Validates input, then delegates to analyze_and_decide().
        """
        if not isinstance(profile, dict):
            logger.error(
                f"[run_ensemble] Expected dict, "
                f"got {type(profile).__name__}: {str(profile)[:200]}"
            )
            profile = {}

        return self.analyze_and_decide(profile, task)

    # ─────────────────────────────────────────────────────────────────────────
    # DATASET PROFILING
    # ─────────────────────────────────────────────────────────────────────────
    def profile_dataset(self, df: pd.DataFrame) -> dict:
        """
        Build a statistical profile dict from a DataFrame.
        Always returns a dict — never raises.
        """
        try:
            numeric_cols     = df.select_dtypes(include="number").columns.tolist()
            categorical_cols = df.select_dtypes(include="object").columns.tolist()

            missing_rate  = float(df.isnull().mean().mean())
            skewness_vals = df[numeric_cols].skew().tolist() if numeric_cols else [0.0]
            mean_skewness = float(sum(skewness_vals) / len(skewness_vals))

            # IQR-based outlier detection
            has_outliers = False
            for col in numeric_cols:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr    = q3 - q1
                if ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).any():
                    has_outliers = True
                    break

            max_cardinality = (
                int(df[categorical_cols].nunique().max())
                if categorical_cols else 0
            )

            result = {
                "n_rows":             len(df),
                "n_cols":             len(df.columns),
                "n_numeric_cols":     len(numeric_cols),
                "n_categorical_cols": len(categorical_cols),
                "missing_rate":       round(missing_rate,  4),
                "mean_skewness":      round(mean_skewness, 4),
                "has_outliers":       has_outliers,
                "max_cardinality":    max_cardinality,
                "numeric_cols":       numeric_cols,
                "categorical_cols":   categorical_cols,
            }

        except Exception as e:
            logger.error(f"[profile_dataset] Profiling failed: {e}")
            result = {
                "n_rows":             len(df),
                "n_cols":             len(df.columns),
                "n_numeric_cols":     0,
                "n_categorical_cols": 0,
                "missing_rate":       0.0,
                "mean_skewness":      0.0,
                "has_outliers":       False,
                "max_cardinality":    0,
                "numeric_cols":       [],
                "categorical_cols":   [],
            }

        # Final type safety check
        assert isinstance(result, dict), \
            f"profile_dataset must return dict, got {type(result)}"
        return result

    def _detect_missing_pattern(self, series: pd.Series) -> str:
        pct = series.isna().mean()
        if pct < 0.02:   return "MCAR"
        elif pct < 0.15: return "MAR_likely"
        else:            return "MNAR_possible"

    # ─────────────────────────────────────────────────────────────────────────
    # PROMPT BUILDER
    # ─────────────────────────────────────────────────────────────────────────
    def _build_prompt(self, profile: dict, task: str) -> str:
        """Format the decision prompt with the dataset profile."""
        profile_json = json.dumps(profile, indent=2, default=str)
        return DECISION_PROMPT_TEMPLATE.format(
            profile_json=profile_json,
            task=task
        )

    # ─────────────────────────────────────────────────────────────────────────
    # RESPONSE PARSER
    # ─────────────────────────────────────────────────────────────────────────
    def _parse_response(self, raw: str, model_name: str) -> LLMDecision:
        """
        Parse raw LLM JSON response into an LLMDecision.
        Returns a safe default LLMDecision on any parse failure.
        """
        try:
            d = self.client.extract_json(raw)
            return self._dict_to_decision(d, model_name, raw, latency=0.0)
        except Exception as e:
            logger.warning(f"[{model_name}] JSON parse failed: {e}")
            return LLMDecision(
                model             = model_name,
                raw_response      = raw,
                reasoning_summary = f"Parse error: {e}",
                confidence        = 0.5,
                latency_ms        = 0.0,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # SINGLE MODEL QUERY
    # ─────────────────────────────────────────────────────────────────────────
    def _query_model(
        self,
        role:    ModelRole,
        profile: dict,
        task:    str,
    ) -> LLMDecision:
        """
        Query one model and return its LLMDecision.
        Falls back to rule-based decision on any error.
        """
        # Guard — validate profile type
        if not isinstance(profile, dict):
            logger.error(
                f"[_query_model:{role.value}] profile must be dict, "
                f"got {type(profile).__name__}. Substituting empty dict."
            )
            profile = {}

        start = time.time()

        try:
            prompt = self._build_prompt(profile, task)
            system = SYSTEM_PROMPTS.get(role, "")

            raw, latency = self.client.generate(
                model_role=role,
                prompt=prompt,
                system=system
            )

            decision            = self._parse_response(raw, role.value)
            decision.latency_ms = round(latency, 2)

            logger.info(f"[{role.value}] ✅ responded in {latency:.0f}ms "
                        f"| confidence={decision.confidence:.0%}")
            return decision

        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.warning(
                f"[{role.value}] ❌ query failed after {latency:.0f}ms: {e}"
            )
            fallback            = self._fallback_decision(role.value, profile)
            fallback.latency_ms = round(latency, 2)
            return fallback

    # ─────────────────────────────────────────────────────────────────────────
    # DICT → LLMDecision
    # ─────────────────────────────────────────────────────────────────────────
    def _dict_to_decision(
        self,
        d:       dict,
        model:   str,
        raw:     str,
        latency: float,
    ) -> LLMDecision:
        return LLMDecision(
            model                 = model,
            imputation_strategy   = d.get("imputation_strategy",      "median"),
            imputation_reasoning  = d.get("imputation_reasoning",      ""),
            knn_neighbors         = int(d.get("knn_neighbors",         5)),
            power_transform       = d.get("power_transform",           "yeo-johnson"),
            outlier_method        = d.get("outlier_method",            "iqr"),
            outlier_threshold     = float(d.get("outlier_threshold",   1.5)),
            correlation_threshold = float(d.get("correlation_threshold", 0.90)),
            feature_engineering   = d.get("feature_engineering", {
                "polynomial":           False,
                "poly_degree":          2,
                "interactions_only":    False,
                "interaction_features": False,
            }),
            encoding = d.get("encoding", {
                "onehot_features":  [],
                "ordinal_features": [],
            }),
            scaler            = d.get("scaler",            "standard"),
            reasoning_summary = d.get("reasoning_summary", ""),
            raw_response      = raw,
            confidence        = float(d.get("confidence",  0.75)),
            latency_ms        = latency,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # GEMMA2 TIEBREAKER
    # ─────────────────────────────────────────────────────────────────────────
    def _tiebreak_with_gemma2(
        self,
        conflicts: list,
        profile:   dict,
        phi4:      LLMDecision,
        mistral:   LLMDecision,
    ) -> tuple[dict, float]:
        """
        Called ONLY when Phi-4 and Mistral disagree on one or more keys.
        Gemma2 receives both opinions + the dataset profile and casts
        the deciding vote.

        Returns:
            resolved_votes : dict  { key: chosen_value }
            latency_ms     : float (-1.0 if Gemma2 itself fails)
        """
        conflict_lines = "\n".join([
            f'  - "{c["key"]}":  '
            f'Phi-4 says "{c["phi4"]}", '
            f'Mistral says "{c["mistral"]}"'
            for c in conflicts
        ])

        profile_json   = json.dumps(profile, indent=2, default=str)
        tiebreak_prompt = TIEBREAK_PROMPT_TEMPLATE.format(
            profile_json   = profile_json,
            conflict_lines = conflict_lines,
        )

        try:
            raw, latency = self.client.generate(
                model_role = ModelRole.GEMMA2,
                prompt     = tiebreak_prompt,
                system     = SYSTEM_PROMPTS[ModelRole.GEMMA2],
            )
            resolved = self.client.extract_json(raw)

            logger.info(
                f"[Gemma2 tiebreaker] ✅ resolved {len(resolved)} "
                f"conflict(s) in {latency:.0f}ms → {resolved}"
            )
            return resolved, round(latency, 2)

        except Exception as e:
            logger.warning(
                f"[Gemma2 tiebreaker] ❌ failed: {e} "
                f"— falling back to Phi-4 decisions"
            )
            # Phi-4 is the last-resort fallback
            return {c["key"]: c["phi4"] for c in conflicts}, -1.0

    # ─────────────────────────────────────────────────────────────────────────
    # PARALLEL ENSEMBLE RUNNER
    # ─────────────────────────────────────────────────────────────────────────
    def analyze_and_decide(
        self, profile: dict, task: str
    ) -> EnsembleDecision:
        """
        Run Phi-4 and Mistral in parallel threads,
        then resolve conflicts (invoking Gemma2 if needed).
        """
        logger.info("[Ensemble] Starting Phi-4 + Mistral parallel query...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            phi4_future    = executor.submit(
                self._query_model, ModelRole.PHI4,    profile, task
            )
            mistral_future = executor.submit(
                self._query_model, ModelRole.MISTRAL, profile, task
            )
            phi4_dec    = phi4_future.result()
            mistral_dec = mistral_future.result()

        logger.info(
            f"[Ensemble] Phi-4={phi4_dec.latency_ms:.0f}ms "
            f"| Mistral={mistral_dec.latency_ms:.0f}ms"
        )

        # Pass profile so Gemma2 tiebreaker can use dataset context
        return self._resolve_ensemble(phi4_dec, mistral_dec, profile)

    # ─────────────────────────────────────────────────────────────────────────
    # ENSEMBLE RESOLVER (with Gemma2 tiebreaker)
    # ─────────────────────────────────────────────────────────────────────────
    def _resolve_ensemble(
        self,
        phi4:    LLMDecision,
        mistral: LLMDecision,
        profile: dict = None,
    ) -> EnsembleDecision:
        """
        Merge Phi-4 and Mistral decisions.
        If any CONFLICT_KEYS differ → invoke Gemma2 tiebreaker.
        """
        if profile is None:
            profile = {}

        # ── Step 1: Detect conflicts ──────────────────────────────────────
        raw_conflicts = []
        for key in self.CONFLICT_KEYS:
            v1, v2 = getattr(phi4, key), getattr(mistral, key)
            if v1 != v2:
                raw_conflicts.append({
                    "key":    key,
                    "phi4":   v1,
                    "mistral": v2,
                })

        agreement_score = round(
            1.0 - len(raw_conflicts) / max(len(self.CONFLICT_KEYS), 1), 3
        )
        tiebreak_used   = len(raw_conflicts) > 0

        logger.info(
            f"[Ensemble] Agreement={agreement_score:.0%} | "
            f"Conflicts={len(raw_conflicts)}/{len(self.CONFLICT_KEYS)} | "
            f"Tiebreak={'YES' if tiebreak_used else 'NO'}"
        )

        # ── Step 2: Ask Gemma2 if conflicts exist ─────────────────────────
        gemma2_votes   = {}
        gemma2_latency = 0.0

        if tiebreak_used:
            gemma2_votes, gemma2_latency = self._tiebreak_with_gemma2(
                conflicts = raw_conflicts,
                profile   = profile,
                phi4      = phi4,
                mistral   = mistral,
            )

        # ── Step 3: Build resolved conflicts list ─────────────────────────
        conflicts = []
        for c in raw_conflicts:
            key      = c["key"]
            resolved = gemma2_votes.get(key, c["phi4"])   # Gemma2 → fallback Phi-4
            method   = "gemma2_tiebreak" if key in gemma2_votes else "phi4_fallback"
            conflicts.append({
                "key":      key,
                "phi4":     c["phi4"],
                "mistral":  c["mistral"],
                "resolved": resolved,
                "method":   method,
            })

        resolved_map = {c["key"]: c["resolved"] for c in conflicts}

        # ── Step 4: Helper resolvers ──────────────────────────────────────
        def resolve_str(key: str) -> str:
            """Use Gemma2 vote for conflicts; unanimous value otherwise."""
            v1, v2 = getattr(phi4, key), getattr(mistral, key)
            if v1 == v2:
                return v1
            return resolved_map.get(key, v1)

        def resolve_float(key: str, w1: float = 0.6, w2: float = 0.4) -> float:
            """Weighted average for numeric fields."""
            return round(
                w1 * getattr(phi4, key) + w2 * getattr(mistral, key), 3
            )

        def merge_encoding() -> dict:
            ohe  = list(set(
                phi4.encoding.get("onehot_features",  []) +
                mistral.encoding.get("onehot_features",  [])
            ))
            ord_ = list(set(
                phi4.encoding.get("ordinal_features", []) +
                mistral.encoding.get("ordinal_features", [])
            ))
            # Avoid overlap — onehot takes priority
            return {
                "onehot_features":  ohe,
                "ordinal_features": [c for c in ord_ if c not in ohe],
            }

        def merge_fe() -> dict:
            f1, f2 = phi4.feature_engineering, mistral.feature_engineering
            return {
                "polynomial": (
                    f1.get("polynomial")          and
                    f2.get("polynomial")
                ),
                "poly_degree": max(
                    f1.get("poly_degree", 2),
                    f2.get("poly_degree", 2)
                ),
                "interactions_only": (
                    f1.get("interactions_only")   and
                    f2.get("interactions_only")
                ),
                "interaction_features": (
                    f1.get("interaction_features") or
                    f2.get("interaction_features")
                ),
            }

        # ── Step 5: Build final merged LLMDecision ────────────────────────
        gemma2_note = ""
        if tiebreak_used:
            if gemma2_latency > 0:
                gemma2_note = (
                    f"\n\n[Gemma2 tiebreaker @ {gemma2_latency:.0f}ms] "
                    f"Resolved {len(gemma2_votes)} conflict(s): {gemma2_votes}"
                )
            else:
                gemma2_note = (
                    "\n\n[Gemma2 tiebreaker] ❌ Failed — "
                    "Phi-4 used as last-resort fallback."
                )

        final = LLMDecision(
            model                = "phi4+mistral+gemma2_ensemble",
            imputation_strategy  = resolve_str("imputation_strategy"),
            imputation_reasoning = (
                f"[Phi-4]   {phi4.imputation_reasoning}\n"
                f"[Mistral] {mistral.imputation_reasoning}"
            ),
            knn_neighbors        = int(round(resolve_float("knn_neighbors"))),
            power_transform      = resolve_str("power_transform"),
            outlier_method       = resolve_str("outlier_method"),
            outlier_threshold    = resolve_float("outlier_threshold"),
            correlation_threshold= resolve_float("correlation_threshold"),
            feature_engineering  = merge_fe(),
            encoding             = merge_encoding(),
            scaler               = resolve_str("scaler"),
            reasoning_summary    = (
                f"[Phi-4   @ {phi4.latency_ms:.0f}ms] {phi4.reasoning_summary}\n\n"
                f"[Mistral @ {mistral.latency_ms:.0f}ms] {mistral.reasoning_summary}"
                f"{gemma2_note}"
            ),
            confidence = round(
                0.6 * phi4.confidence + 0.4 * mistral.confidence, 3
            ),
            latency_ms = max(
                phi4.latency_ms,
                mistral.latency_ms,
                gemma2_latency if tiebreak_used else 0.0,
            ),
        )

        return EnsembleDecision(
            final            = final,
            phi4_decision    = phi4,
            mistral_decision = mistral,
            conflicts        = conflicts,
            agreement_score  = agreement_score,
            tiebreak_used    = tiebreak_used,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # RULE-BASED FALLBACK (no LLM needed)
    # ─────────────────────────────────────────────────────────────────────────
    def get_fallback_decisions(self, profile: dict) -> EnsembleDecision:
        """Public helper — returns a full EnsembleDecision using rules only."""
        dec = self._fallback_decision("rule_based", profile)
        return EnsembleDecision(
            final            = dec,
            phi4_decision    = dec,
            mistral_decision = dec,
            conflicts        = [],
            agreement_score  = 1.0,
            tiebreak_used    = False,
        )

    def _fallback_decision(self, model: str, profile: dict) -> LLMDecision:
        """
        Deterministic rule-based decision when LLM is unavailable.
        latency_ms = -1.0 signals fallback was used (not a real query).
        """
        if not isinstance(profile, dict):
            logger.error(
                f"[_fallback_decision] Expected dict, "
                f"got {type(profile).__name__}: {str(profile)[:200]}"
            )
            profile = {}

        # Read profile safely
        missing_rate    = profile.get("missing_rate",       0.0)
        skewness        = profile.get("mean_skewness",      0.0)
        has_outliers    = profile.get("has_outliers",       False)
        n_rows          = profile.get("n_rows",             1000)
        n_numeric       = profile.get("n_numeric_cols",     0)
        n_cat_cols      = profile.get("n_categorical_cols", 0)
        max_cardinality = profile.get("max_cardinality",    10)

        # ── Imputation ────────────────────────────────────────────────────
        if   missing_rate == 0.0:  strategy = "none"
        elif missing_rate < 0.05:  strategy = "median"
        elif missing_rate < 0.20:  strategy = "knn"
        else:                      strategy = "iterative"

        # ── Transform ─────────────────────────────────────────────────────
        if   abs(skewness) < 0.5:  transform = "none"
        elif abs(skewness) < 2.0:  transform = "yeo-johnson"
        else:                      transform = "log1p"

        # ── Scaler ────────────────────────────────────────────────────────
        scaler = "robust" if has_outliers else "standard"

        # ── Outlier method ────────────────────────────────────────────────
        if   n_rows < 500:   outlier_method = "iqr"
        elif n_rows < 5000:  outlier_method = "zscore"
        else:                outlier_method = "isolation_forest"

        # ── Encoding ──────────────────────────────────────────────────────
        encoding = {
            "onehot_features":  [],
            "ordinal_features": [],
        }

        # ── Feature engineering ───────────────────────────────────────────
        feature_engineering = {
            "polynomial":           n_numeric <= 10,
            "poly_degree":          2,
            "interactions_only":    False,
            "interaction_features": n_numeric <= 15,
        }

        return LLMDecision(
            model                 = model,
            imputation_strategy   = strategy,
            power_transform       = transform,
            scaler                = scaler,
            outlier_method        = outlier_method,
            outlier_threshold     = 1.5,
            correlation_threshold = 0.90,
            feature_engineering   = feature_engineering,
            encoding              = encoding,
            confidence            = 0.70,
            latency_ms            = -1.0,   # ← signals fallback, not a real query
            reasoning_summary     = (
                f"Rule-based fallback applied. "
                f"[missing={missing_rate:.0%}, "
                f"skew={skewness:.2f}, "
                f"outliers={has_outliers}, "
                f"rows={n_rows}]"
            ),
        )
   
   
    # ✅ ADD interpret() RIGHT HERE — still inside the class
    def interpret(self, pipeline_output: dict) -> str:
        """
        Generate a plain-English summary of pipeline results.
        Falls back gracefully if Ollama is unavailable.
        """
        try:
            summary_prompt = (
                f"Summarize these ML pipeline results in 3 sentences:\n"
                f"Task: {pipeline_output.get('task', 'unknown')}\n"
                f"Models run: {list(pipeline_output.get('results', {}).keys())}\n"
                f"Comparison:\n"
                f"{pipeline_output.get('comparison', 'N/A')}"
            )
            raw, _ = self.client.generate(
                model_role=ModelRole.PHI4,
                prompt=summary_prompt,
                system="You are a concise data science reporter. Summarize results clearly.",
            )
            return raw.strip()
        except Exception as e:
            logger.warning(f"[interpret] LLM summary failed: {e}")
            return (
                f"Pipeline completed successfully. "
                f"Task: {pipeline_output.get('task', 'unknown')}. "
                f"Models evaluated: "
                f"{list(pipeline_output.get('results', {}).keys())}."
            )  
