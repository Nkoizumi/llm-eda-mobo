# run_feedback.py
#
# Quick-start script to run the full LLM feedback pipeline.
#
# Usage:
#   python run_feedback.py                        ← rule-based mode
#   python run_feedback.py --llm                  ← LLM debate mode
#   python run_feedback.py --llm --csv data.csv   ← custom dataset

import argparse
import json
import time
import pandas as pd
from pathlib import Path

from feedback_controller import FeedbackConfig, AutoEDAFeedbackController


# ──────────────────────────────────────────────────────────────────
# CLI Arguments
# ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="AutoEDA Feedback Loop — Quick Start Runner"
    )

    parser.add_argument(
        "--csv",
        type    = str,
        default = None,
        help    = "Path to input CSV file. Uses synthetic data if not provided."
    )
    parser.add_argument(
        "--target",
        type    = str,
        default = "target",
        help    = "Name of the target column (default: 'target')"
    )
    parser.add_argument(
        "--task",
        type    = str,
        default = "regression",
        choices = ["regression", "classification"],
        help    = "Task type (default: regression)"
    )
    parser.add_argument(
        "--threshold",
        type    = float,
        default = 0.85,
        help    = "Performance threshold to stop early (default: 0.85)"
    )
    parser.add_argument(
        "--iterations",
        type    = int,
        default = 5,
        help    = "Max feedback iterations (default: 5)"
    )
    parser.add_argument(
        "--llm",
        action  = "store_true",
        default = False,
        help    = "Enable LLM debate pipeline (default: rule-based only)"
    )
    parser.add_argument(
        "--ollama-host",
        type    = str,
        default = "http://localhost:11434",
        help    = "Ollama host URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--agent-a",
        type    = str,
        default = "phi4",
        help    = "Debate Agent A model (default: phi4)"
    )
    parser.add_argument(
        "--agent-b",
        type    = str,
        default = "mistral",
        help    = "Debate Agent B model (default: mistral)"
    )
    parser.add_argument(
        "--arbitrator",
        type    = str,
        default = "phi4",
        help    = "Arbitrator model (default: phi4)"
    )
    parser.add_argument(
        "--timeout",
        type    = int,
        default = 120,
        help    = "LLM request timeout in seconds (default: 120)"
    )
    parser.add_argument(
        "--output",
        type    = str,
        default = "results/",
        help    = "Output directory for results (default: results/)"
    )
    parser.add_argument(
        "--quiet",
        action  = "store_true",
        default = False,
        help    = "Suppress verbose output"
    )

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────
# Synthetic Data Generator (Fallback if no CSV provided)
# ──────────────────────────────────────────────────────────────────

def generate_synthetic_data(task_type: str, n_samples: int = 1000) -> pd.DataFrame:
    """
    Generates a synthetic dataset with intentional issues:
        - Ghost feature  (employment_years ≈ age * 0.5)
        - Outliers       (loan_amount has extreme values)
        - Right skew     (credit_score — log transform candidate)
        - Clean feature  (income — bounded, MinMaxScaler candidate)
    """
    import numpy as np
    rng = np.random.default_rng(42)

    n = n_samples

    age               = rng.integers(22, 65,  size=n).astype(float)
    employment_years  = age * 0.5 + rng.normal(0, 1, n)           # Ghost feature
    income            = rng.uniform(20_000, 200_000, size=n)       # Clean, bounded
    loan_amount       = rng.exponential(scale=15_000, size=n)      # Outlier-heavy
    credit_score      = rng.exponential(scale=300, size=n) + 300   # Right-skewed

    # Inject outliers into loan_amount
    outlier_idx = rng.choice(n, size=int(n * 0.03), replace=False)
    loan_amount[outlier_idx] *= 20

    if task_type == "regression":
        target = (
            0.4 * income / 1000
            + 0.3 * credit_score / 100
            - 0.2 * loan_amount / 10_000
            + rng.normal(0, 2, n)
        )
    else:
        target = (
            (income > 80_000) &
            (credit_score > 400) &
            (loan_amount < 30_000)
        ).astype(int)

    df = pd.DataFrame({
        "age"              : age,
        "employment_years" : employment_years,
        "income"           : income,
        "loan_amount"      : loan_amount,
        "credit_score"     : credit_score,
        "target"           : target
    })

    return df


# ──────────────────────────────────────────────────────────────────
# Results Writer
# ──────────────────────────────────────────────────────────────────

def save_results(
    output_dir:    str,
    result_df:     pd.DataFrame,
    history:       list,
    model_name:    str,
    elapsed:       float,
    use_llm:       bool
):
    """
    Saves:
        - processed_data.csv   — final corrected DataFrame
        - history.json         — full iteration history
        - summary.txt          — human-readable run summary
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Save processed DataFrame ─────────────────────────────
    csv_path = out / "processed_data.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"\n💾 Processed data saved → {csv_path}")

    # ── Save full history ────────────────────────────────────
    history_path = out / "history.json"
    with open(history_path, "w") as f:
        # EDA reports may not be JSON serializable — convert safely
        safe_history = []
        for entry in history:
            safe_entry = {}
            for k, v in entry.items():
                try:
                    json.dumps(v)
                    safe_entry[k] = v
                except (TypeError, ValueError):
                    safe_entry[k] = str(v)
            safe_history.append(safe_entry)
        json.dump(safe_history, f, indent=2)
    print(f"📜 Iteration history saved → {history_path}")

    # ── Save summary ─────────────────────────────────────────
    summary_path = out / "summary.txt"
    final_score  = history[-1]["score"] if history else "N/A"
    best_score   = max(e["score"] for e in history) if history else "N/A"

    with open(summary_path, "w") as f:
        f.write("=" * 52 + "\n")
        f.write("  AutoEDA Feedback Loop — Run Summary\n")
        f.write("=" * 52 + "\n")
        f.write(f"  Mode          : {'LLM Debate' if use_llm else 'Rule-Based'}\n")
        f.write(f"  Best Model    : {model_name or 'N/A'}\n")
        f.write(f"  Iterations    : {len(history)}\n")
        f.write(f"  Final Score   : {final_score:.4f}\n")
        f.write(f"  Best Score    : {best_score:.4f}\n")
        f.write(f"  Elapsed Time  : {elapsed:.2f}s\n")
        f.write("\nIteration Log:\n")
        for entry in history:
            f.write(
                f"  [{entry['iteration']}] "
                f"score={entry['score']:.4f} | "
                f"model={entry['model']} | "
                f"corrections={entry.get('corrections', 'none')}\n"
            )

    print(f"📝 Summary saved         → {summary_path}")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Banner ───────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("  🤖 AutoEDA Feedback Loop — Quick Start Runner")
    print("=" * 52)
    print(f"  Mode       : {'🧠 LLM Debate' if args.llm else '📐 Rule-Based'}")
    print(f"  Task       : {args.task}")
    print(f"  Threshold  : {args.threshold}")
    print(f"  Iterations : {args.iterations}")
    if args.llm:
        print(f"  Agent A    : {args.agent_a}")
        print(f"  Agent B    : {args.agent_b}")
        print(f"  Arbitrator : {args.arbitrator}")
        print(f"  Ollama     : {args.ollama_host}")
    print("=" * 52 + "\n")

    # ── Load or Generate Data ────────────────────────────────
    if args.csv:
        print(f"📂 Loading dataset: {args.csv}")
        raw_df = pd.read_csv(args.csv)
        if args.target not in raw_df.columns:
            raise ValueError(
                f"Target column '{args.target}' not found in CSV. "
                f"Available columns: {list(raw_df.columns)}"
            )
        print(f"   Rows: {len(raw_df):,} | Cols: {len(raw_df.columns)}")
    else:
        print("⚗️  No CSV provided — generating synthetic dataset...")
        raw_df = generate_synthetic_data(
            task_type = args.task,
            n_samples = 1000
        )
        print(
            f"   Rows: {len(raw_df):,} | "
            f"Cols: {len(raw_df.columns)} | "
            f"Features: age, employment_years, income, loan_amount, credit_score"
        )

    # ── Build Config ─────────────────────────────────────────
    config = FeedbackConfig(
        task_type        = args.task,
        metric_threshold = args.threshold,
        max_iterations   = args.iterations,
        target_col       = args.target,
        verbose          = not args.quiet,
        use_local_llm    = args.llm,
        ollama_host      = args.ollama_host,
        debate_agent_a   = args.agent_a,
        debate_agent_b   = args.agent_b,
        arbitrator_model = args.arbitrator,
        llm_timeout      = args.timeout
    )

    # ── Run Feedback Loop ────────────────────────────────────
    controller   = AutoEDAFeedbackController(config)
    start_time   = time.time()

    result_df, model_name, history = controller.run(raw_df)

    elapsed = time.time() - start_time

    # ── Print Final Summary ──────────────────────────────────
    print("\n" + "=" * 52)
    print("  ✅ Run Complete")
    print("=" * 52)

    final_score = history[-1]["score"] if history else 0.0
    best_score  = max(e["score"] for e in history) if history else 0.0

    print(f"  Best Model   : {model_name or 'N/A'}")
    print(f"  Iterations   : {len(history)}")
    print(f"  Final Score  : {final_score:.4f}")
    print(f"  Best Score   : {best_score:.4f}")
    print(f"  Elapsed      : {elapsed:.2f}s")

    print("\n  📊 Iteration Breakdown:")
    for entry in history:
        print(
            f"    [{entry['iteration']}] "
            f"score={entry['score']:.4f} | "
            f"model={entry['model']} | "
            f"corrections={entry.get('corrections', 'none')}"
        )

    # ── Save Results ─────────────────────────────────────────
    save_results(
        output_dir = args.output,
        result_df  = result_df,
        history    = history,
        model_name = model_name,
        elapsed    = elapsed,
        use_llm    = args.llm
    )


# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()

