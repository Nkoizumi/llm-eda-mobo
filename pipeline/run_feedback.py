import pandas as pd
from feedback_controller import AutoEDAFeedbackController, FeedbackConfig

# --- Configuration ---
config = FeedbackConfig(
    task_type="regression",        # or "classification"
    metric_threshold=0.95,
    max_iterations=5,
    models=["xgboost", "neural_network"],
    verbose=True,
)

# --- Load Data ---
df = pd.read_csv("your_dataset.csv")

# --- Run Feedback Loop ---
controller = AutoEDAFeedbackController(config)
final_df, best_model, history = controller.run(df)

# --- View History ---
print("\n📈 Performance History:")
for h in history:
    metric = "R²" if config.task_type == "regression" else "Accuracy"
    print(f"  Iter {h['iteration']} | {h['model']} | {metric}: {h['score']:.4f}")
