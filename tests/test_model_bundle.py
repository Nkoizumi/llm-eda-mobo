"""Testing a trained model on new data (pipeline/model_bundle.py).

The property that matters most: test data goes through preprocessing FITTED ON
THE TRAINING ROWS. Refitting the imputer / scaler / encoders on the test file
would let the test set shape its own preprocessing, and would make a row's
prediction depend on which other rows happen to be in the same test file.
"""
from __future__ import annotations

import json
import zipfile
import io

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge

from conftest import DATA
from pipeline.model_bundle import (
    ModelBundle,
    align_labels,
    classification_metrics,
    fit_single_target_bundle,
)
from pipeline.orchestrator import AutoEDAPipeline


def _bundle(df, target, task, est, target_cols=None):
    target_cols = target_cols or [target]
    eda = AutoEDAPipeline(target_col=target, task=task, use_local_llm=False)
    eda.build_pipeline(df.drop(columns=target_cols))
    return fit_single_target_bundle(
        df=df, preprocessor=eda.pipeline_, estimator=est, target=target,
        target_cols=target_cols, task=task, estimator_name=type(est).__name__,
        cv_metrics={target: {"r2": 0.9}}, hpo={"best_params": {"alpha": 0.5},
                                               "best_value": 0.91},
    )


@pytest.fixture(scope="module")
def fish():
    df = pd.read_csv(DATA / "Fish.csv")
    df.loc[::7, "Height"] = np.nan                  # exercise the imputer
    rng = np.random.default_rng(0)
    test_mask = rng.random(len(df)) < 0.25
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


@pytest.fixture(scope="module")
def fish_bundle(fish):
    train, _ = fish
    return _bundle(train, "Weight", "regression",
                   RandomForestRegressor(n_estimators=30, random_state=0))


def test_predictions_match_a_manual_train_fit_test_transform(fish, fish_bundle):
    train, test = fish
    eda = AutoEDAPipeline(target_col="Weight", task="regression", use_local_llm=False)
    eda.build_pipeline(train.drop(columns=["Weight"]))
    prep = eda.pipeline_
    Xtr = prep.fit_transform(train.drop(columns=["Weight"]))
    est = RandomForestRegressor(n_estimators=30, random_state=0).fit(
        Xtr, train["Weight"].to_numpy(np.float32))
    expected = est.predict(prep.transform(test.drop(columns=["Weight"])).astype(np.float32))

    got = fish_bundle.predict(test)["Weight"].to_numpy()

    np.testing.assert_allclose(got, expected, rtol=1e-5)


def test_a_row_is_predicted_the_same_alone_or_in_a_batch(fish, fish_bundle):
    """Fails if anything is refit on the test data."""
    _, test = fish
    batch = fish_bundle.predict(test)["Weight"].to_numpy()
    alone = np.array([fish_bundle.predict(test.iloc[[i]])["Weight"].iloc[0]
                      for i in range(len(test))])
    np.testing.assert_allclose(alone, batch, rtol=1e-6)


def test_imputer_uses_training_medians(fish, fish_bundle):
    train, _ = fish
    rec = fish_bundle.preprocessing_record()
    num = next(g for g in rec["pipeline"]["steps"][0]["groups"] if g["name"] == "num")
    imputer = next(s for s in num["steps"] if s["name"] == "imputer")
    assert imputer["statistics_"]["Height"] == pytest.approx(train["Height"].median())


def test_column_order_and_target_absence_do_not_matter(fish, fish_bundle):
    _, test = fish
    ref = fish_bundle.predict(test)["Weight"].to_numpy()
    shuffled = test[list(reversed(test.columns))].drop(columns=["Weight"])
    np.testing.assert_allclose(fish_bundle.predict(shuffled)["Weight"].to_numpy(), ref)


def test_missing_required_column_is_reported_and_raises(fish, fish_bundle):
    _, test = fish
    bad = test.drop(columns=["Length1"])
    assert "Length1" in fish_bundle.check_test_frame(bad)["missing"]
    with pytest.raises(KeyError, match="Length1"):
        fish_bundle.predict(bad)


def test_checks_flag_extrapolation_and_unseen_categories(fish, fish_bundle):
    _, test = fish
    odd = test.head(3).copy()
    odd.loc[0, "Width"] = 1e6
    odd.loc[1, "Species"] = "Shark"
    chk = fish_bundle.check_test_frame(odd)
    assert chk["out_of_range"] == {"Width": 1}
    assert chk["rows_outside_training_range"].tolist() == [True, False, False]
    assert chk["unseen_levels"] == {"Species": ["Shark"]}
    fish_bundle.predict(odd)          # unseen category must not crash


def test_zip_round_trip_and_json_records(fish, fish_bundle):
    _, test = fish
    data = fish_bundle.to_zip_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert set(zf.namelist()) == {"model.joblib", "preprocessing.json",
                                      "best_params.json", "manifest.json"}
        params = json.loads(zf.read("best_params.json"))
        prep = json.loads(zf.read("preprocessing.json"))
    assert params["hpo_best_params"] == {"alpha": 0.5}
    assert params["final_estimator_params"]["n_estimators"] == 30
    assert prep["required_input_columns"]

    loaded = ModelBundle.from_zip_bytes(data)
    pd.testing.assert_frame_equal(loaded.predict(test), fish_bundle.predict(test))


def test_a_tampered_bundle_is_rejected(fish_bundle):
    src = zipfile.ZipFile(io.BytesIO(fish_bundle.to_zip_bytes()))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in src.namelist():
            blob = src.read(name)
            zf.writestr(name, blob + b"x" if name == "model.joblib" else blob)
    with pytest.raises(ValueError, match="hash"):
        ModelBundle.from_zip_bytes(buf.getvalue())


def test_save_dir_writes_all_files(tmp_path, fish_bundle):
    out = fish_bundle.save_dir(tmp_path)
    assert sorted(p.name for p in out.iterdir()) == [
        "best_params.json", "manifest.json", "model.joblib", "preprocessing.json"]


def test_classification_returns_original_labels():
    df = pd.read_csv(DATA / "Titanic-Dataset.csv")[
        ["Survived", "Pclass", "Sex", "Age", "Fare", "Embarked"]]
    df["Survived"] = df["Survived"].map({0: "died", 1: "survived"})
    train, test = df.iloc[:600], df.iloc[600:]
    b = _bundle(train, "Survived", "classification",
                RandomForestClassifier(n_estimators=30, random_state=0))
    pred = b.predict(test)["Survived"]
    assert set(pred) <= {"died", "survived"}
    m = classification_metrics(test["Survived"], pred)
    assert 0.6 < m["accuracy"] <= 1.0


def test_align_labels_handles_int_vs_float_csv_columns():
    t, p = align_labels(np.array([0.0, 1.0]), np.array(["0", "1"], dtype=object))
    assert (t == p).all()


def test_rows_with_missing_target_are_left_out_of_training(fish):
    train, _ = fish
    train = train.copy()
    train.loc[:9, "Weight"] = np.nan
    b = _bundle(train, "Weight", "regression", Ridge())
    assert b.train_summary["n_rows"] == len(train) - 10


def test_project_wrappers_bundle_and_reload(fish):
    """XGBoost / NN wrappers are not sklearn estimators; they must still be
    copied, fitted, saved on CPU and reloaded."""
    pytest.importorskip("xgboost")
    pytest.importorskip("torch")
    from pipeline.models.neural_network_model import NeuralNetworkModel
    from pipeline.models.xgboost_model import XGBoostModel

    train, test = fish
    for est in (XGBoostModel(task="regression", hpo_params={"n_estimators": 20}),
                NeuralNetworkModel(task="regression", epochs=5, patience=2)):
        b = _bundle(train, "Weight", "regression", est)
        loaded = ModelBundle.from_zip_bytes(b.to_zip_bytes())
        np.testing.assert_allclose(loaded.predict(test)["Weight"],
                                   b.predict(test)["Weight"], rtol=1e-5)
        assert b.best_params_record()["final_estimator_params"]
