import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor


class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def _get_models(self):
        if self.config.task_type == "regression":
            return {
                "xgboost": XGBRegressor(n_estimators=300, learning_rate=0.05,
                                         random_state=42, verbosity=0),
                "neural_network": MLPRegressor(hidden_layer_sizes=(128, 64),
                                                max_iter=500, random_state=42),
            }
        else:
            return {
                "xgboost": XGBClassifier(n_estimators=300, learning_rate=0.05,
                                          use_label_encoder=False, eval_metric="logloss",
                                          random_state=42, verbosity=0),
                "neural_network": MLPClassifier(hidden_layer_sizes=(128, 64),
                                                 max_iter=500, random_state=42),
            }

    def train_and_evaluate(self, df) -> tuple[float, str]:
        """Returns best (score, model_name) across all models."""
        from sklearn.model_selection import train_test_split

        target_col = df.columns[-1]  # assume last col is target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scoring = "r2" if self.config.task_type == "regression" else "accuracy"
        best_score = -np.inf
        best_model_name = None

        for name, model in self._get_models().items():
            if name not in self.config.models:
                continue
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                if self.config.task_type == "regression":
                    score = r2_score(y_test, preds)
                else:
                    score = accuracy_score(y_test, preds)

                if score > best_score:
                    best_score = score
                    best_model_name = name
            except Exception as e:
                print(f"  ⚠️ Model {name} failed: {e}")

        return best_score, best_model_name
