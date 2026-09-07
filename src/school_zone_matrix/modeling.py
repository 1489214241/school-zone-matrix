from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import KEYS, hash_bucket, require_columns
from .features import anchor_features, predictor_columns, static_features, training_anchor_features

MODEL_SPECS = [
    {"name": "combined_logistic", "kind": "logistic", "feature_set": "combined"},
    {"name": "combined_histgb", "kind": "histgb", "feature_set": "combined"},
    {"name": "anchor_logistic", "kind": "anchor_logistic", "feature_set": "anchor"},
]


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def metrics(truth, prediction) -> dict[str, float | int]:
    truth, prediction = np.asarray(truth, int), np.asarray(prediction, int)
    tp = int(((truth == 1) & (prediction == 1)).sum())
    fp = int(((truth == 0) & (prediction == 1)).sum())
    fn = int(((truth == 1) & (prediction == 0)).sum())
    tn = int(((truth == 0) & (prediction == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": precision, "recall": recall, "F1": f1}


def merge_labels(candidates: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    require_columns(labels, KEYS + ["label", "observed"], "labels")
    if labels.duplicated(KEYS).any():
        raise ValueError("labels contains duplicate relation keys")
    data = candidates.merge(labels[KEYS + ["label", "observed"]], on=KEYS, how="left", validate="one_to_one")
    data["observed"] = data.observed.fillna(False).astype(bool)
    data["label"] = pd.to_numeric(data.label, errors="coerce")
    data.loc[~data.observed, "label"] = np.nan
    if not data.loc[data.observed, "label"].isin([0, 1]).all():
        raise ValueError("observed labels must be binary")
    return data


def make_model(specification: dict):
    if specification["kind"] == "histgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=160,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=20260908,
        )
    class_weight = "balanced" if specification["kind"] == "anchor_logistic" else None
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight=class_weight, max_iter=2000, random_state=20260908),
    )


def fit_model(specification: dict, features: pd.DataFrame, data: pd.DataFrame, indices):
    indices = pd.Index(indices)
    observed = data.loc[indices, "observed"].to_numpy(bool)
    truth = data.loc[indices, "label"].to_numpy()[observed].astype(int)
    model = make_model(specification)
    if specification["kind"] == "histgb":
        model.fit(features.iloc[observed], truth, sample_weight=np.where(truth == 1, 8.0, 1.0))
    else:
        model.fit(features.iloc[observed], truth)
    return model


def positive_probability(model, features: pd.DataFrame) -> np.ndarray:
    probability = model.predict_proba(features)
    classes = list(model.classes_)
    return probability[:, classes.index(1)] if 1 in classes else np.zeros(len(features))


def feature_matrix(static: pd.DataFrame, anchor: pd.DataFrame, indices, feature_set: str) -> pd.DataFrame:
    indices = pd.Index(indices)
    if feature_set == "anchor":
        return anchor.loc[indices].astype(np.float32)
    return pd.concat([static.loc[indices], anchor.loc[indices]], axis=1).astype(np.float32)


def decode(frame: pd.DataFrame, probability, threshold: float, rank_cap: int, union_baseline: bool) -> np.ndarray:
    work = frame[["community_id", "baseline"]].copy()
    work["probability"] = np.asarray(probability)
    rank = work.groupby("community_id").probability.rank(method="first", ascending=False)
    prediction = (work.probability.ge(threshold) & rank.le(rank_cap)).to_numpy(int)
    if union_baseline:
        prediction = np.maximum(prediction, work.baseline.to_numpy(int))
    return prediction


def select_decoder(frame: pd.DataFrame, probability) -> tuple[dict, list[dict]]:
    observed = frame.observed.to_numpy(bool)
    truth = frame.label.to_numpy()[observed]
    rows = []
    for union_baseline in [False, True]:
        for rank_cap in [1, 2, 3, 4]:
            for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
                prediction = decode(frame, probability, threshold, rank_cap, union_baseline)
                score = metrics(truth, prediction[observed])
                feasible = score["precision"] >= 0.8 and score["recall"] >= 0.8
                rank = (feasible, min(score["precision"], score["recall"]), score["F1"], score["precision"], -abs(threshold - 0.5))
                rows.append({"threshold": float(threshold), "rank_cap": rank_cap, "union_baseline": union_baseline, "feasible_80_80": feasible, "_rank": rank, **score})
    best = max(rows, key=lambda row: row["_rank"]).copy()
    for row in rows:
        del row["_rank"]
    del best["_rank"]
    return best, rows


def _build_static(data: pd.DataFrame, numeric: list[str], schools: list[str]) -> pd.DataFrame:
    return static_features(data, numeric, schools)


def _outer_indices(data: pd.DataFrame, fold: int):
    test = data.index[data.outer_fold.eq(fold)]
    pool = data.index[data.outer_fold.ne(fold)]
    if set(data.loc[pool, "group_id"]) & set(data.loc[test, "group_id"]):
        raise AssertionError("outer spatial group leakage")
    return pool, test


def crossfit_predict(candidates: pd.DataFrame, labels: pd.DataFrame, output_dir: str | Path, inner_folds: int = 4) -> Path:
    """Write predictions without computing test metrics; evaluation is a separate command."""
    require_columns(candidates, KEYS + ["community_id", "school_id", "group_id", "outer_fold", "x", "y", "baseline"], "candidates")
    output_dir = Path(output_dir)
    if (output_dir / "all_predictions_frozen.json").exists():
        raise RuntimeError("predictions are already frozen")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = merge_labels(candidates, labels)
    numeric = predictor_columns(candidates)
    schools = sorted(map(str, candidates.school_id.unique()))
    static = _build_static(data, numeric, schools)
    parts = []
    for fold in sorted(map(int, data.outer_fold.unique())):
        pool, test = _outer_indices(data, fold)
        working = data.copy()
        working.loc[test, ["label", "observed"]] = [np.nan, False]
        if working.loc[test, "observed"].any():
            raise AssertionError("test labels were not masked")
        inner = working.loc[pool, "group_id"].map(lambda value: hash_bucket(value, inner_folds, f"inner-{fold}-v1"))
        oof = {spec["name"]: pd.Series(np.nan, index=pool) for spec in MODEL_SPECS}
        for inner_fold in range(inner_folds):
            validation = pool[inner.eq(inner_fold)]
            train = pool[inner.ne(inner_fold)]
            if not len(validation):
                continue
            anchor_train = training_anchor_features(working, train, f"anchor-{fold}-{inner_fold}")
            anchor_validation = anchor_features(working, train, validation)
            for specification in MODEL_SPECS:
                train_features = feature_matrix(static, anchor_train, train, specification["feature_set"])
                validation_features = feature_matrix(static, anchor_validation, validation, specification["feature_set"])
                model = fit_model(specification, train_features, working, train)
                oof[specification["name"]].loc[validation] = positive_probability(model, validation_features)
        choices = {}
        selected = None
        for specification in MODEL_SPECS:
            probability = oof[specification["name"]]
            if probability.isna().any():
                raise AssertionError("incomplete pooled calibration predictions")
            decoder, _ = select_decoder(working.loc[pool], probability.to_numpy())
            choices[specification["name"]] = decoder
            rank = (decoder["feasible_80_80"], min(decoder["precision"], decoder["recall"]), decoder["F1"], decoder["precision"])
            if selected is None or rank > selected[0]:
                selected = (rank, specification)
        assert selected is not None
        specification = selected[1]
        decoder = choices[specification["name"]]
        anchor_pool = training_anchor_features(working, pool, f"anchor-final-{fold}")
        anchor_test = anchor_features(working, pool, test)
        pool_features = feature_matrix(static, anchor_pool, pool, specification["feature_set"])
        test_features = feature_matrix(static, anchor_test, test, specification["feature_set"])
        model = fit_model(specification, pool_features, working, pool)
        probability = positive_probability(model, test_features)
        prediction = decode(working.loc[test], probability, decoder["threshold"], decoder["rank_cap"], decoder["union_baseline"])
        columns = [column for column in KEYS + ["community_name", "school_name", "group_id", "outer_fold", "baseline"] if column in data]
        result = data.loc[test, columns].copy()
        result["model"] = specification["name"]
        result["probability"] = probability
        result["prediction"] = prediction
        result["threshold"] = decoder["threshold"]
        result["rank_cap"] = decoder["rank_cap"]
        parts.append(result)
        fold_dir = output_dir / f"outer_{fold}"
        fold_dir.mkdir()
        result.to_csv(fold_dir / "predictions.csv", index=False)
        joblib.dump(model, fold_dir / "model.joblib")
        save_json(fold_dir / "selection.json", {"model": specification, "decoder": decoder, "all_models": choices, "test_labels_masked": True})
    predictions = pd.concat(parts, ignore_index=True).sort_values(KEYS)
    if len(predictions) != len(candidates) or predictions.duplicated(KEYS).any():
        raise AssertionError("frozen predictions do not cover the complete candidate universe")
    prediction_path = output_dir / "predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    save_json(output_dir / "all_predictions_frozen.json", {"utc": datetime.now(timezone.utc).isoformat(), "rows": len(predictions), "prediction_sha256": sha256(prediction_path), "gold_scored": False})
    return prediction_path


def evaluate_frozen(prediction_path: str | Path, labels: pd.DataFrame, output_dir: str | Path) -> dict:
    prediction_path, output_dir = Path(prediction_path), Path(output_dir)
    receipt_path = prediction_path.parent / "all_predictions_frozen.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if sha256(prediction_path) != receipt["prediction_sha256"]:
        raise ValueError("prediction hash does not match the freeze receipt")
    if (output_dir / "evaluation.json").exists():
        raise RuntimeError("this frozen prediction has already been evaluated")
    predictions = pd.read_csv(prediction_path)
    data = predictions.merge(labels[KEYS + ["label", "observed"]], on=KEYS, how="left", validate="one_to_one")
    data = data[data.observed.fillna(False).astype(bool)]
    overall = metrics(data.label, data.prediction)
    rows = [{"evaluation": "overall", **overall}]
    for fold, part in data.groupby("outer_fold"):
        rows.append({"evaluation": f"outer_{int(fold)}", **metrics(part.label, part.prediction)})
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    report = {"classification": "SUPERVISED_DEVELOPMENT", "prediction_sha256": receipt["prediction_sha256"], "overall": overall, "both_at_least_80": overall["precision"] >= 0.8 and overall["recall"] >= 0.8}
    save_json(output_dir / "evaluation.json", report)
    return report


@dataclass
class ReleaseBundle:
    model: object
    specification: dict
    decoder: dict
    reference: pd.DataFrame
    numeric_columns: list[str]
    school_ids: list[str]
    feature_columns: list[str]


def train_release(candidates: pd.DataFrame, labels: pd.DataFrame, model_path: str | Path) -> ReleaseBundle:
    data = merge_labels(candidates, labels)
    # ``DataFrameGroupBy.observed`` is also a pandas configuration attribute,
    # so attribute access does not reliably select the column named observed.
    observed_communities = data.groupby("community_id")["observed"].any()
    if not observed_communities.all():
        raise ValueError("release training requires labels for every reference community")
    indices = data.index
    numeric = predictor_columns(candidates)
    schools = sorted(map(str, candidates.school_id.unique()))
    static = static_features(data, numeric, schools)
    fold = data.group_id.map(lambda value: hash_bucket(value, 4, "release-calibration-v1"))
    probabilities = {spec["name"]: pd.Series(np.nan, index=indices) for spec in MODEL_SPECS}
    for inner in range(4):
        validation, train = indices[fold.eq(inner)], indices[fold.ne(inner)]
        anchor_train = training_anchor_features(data, train, f"release-anchor-{inner}")
        anchor_validation = anchor_features(data, train, validation)
        for specification in MODEL_SPECS:
            xt = feature_matrix(static, anchor_train, train, specification["feature_set"])
            xv = feature_matrix(static, anchor_validation, validation, specification["feature_set"])
            model = fit_model(specification, xt, data, train)
            probabilities[specification["name"]].loc[validation] = positive_probability(model, xv)
    choices = []
    for specification in MODEL_SPECS:
        decoder, _ = select_decoder(data, probabilities[specification["name"]])
        rank = (decoder["feasible_80_80"], min(decoder["precision"], decoder["recall"]), decoder["F1"])
        choices.append((rank, specification, decoder))
    _, specification, decoder = max(choices, key=lambda item: item[0])
    anchor = training_anchor_features(data, indices, "release-final-anchor")
    features = feature_matrix(static, anchor, indices, specification["feature_set"])
    model = fit_model(specification, features, data, indices)
    reference_columns = KEYS + ["community_name", "family_name", "group_id", "x", "y", "label", "observed"]
    bundle = ReleaseBundle(model, specification, decoder, data[reference_columns].copy(), numeric, schools, list(features.columns))
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


def predict_new(bundle: ReleaseBundle, candidates: pd.DataFrame) -> pd.DataFrame:
    if set(candidates.community_id) & set(bundle.reference.community_id):
        raise ValueError("prediction communities overlap release training communities")
    static = static_features(candidates, bundle.numeric_columns, bundle.school_ids)
    combined = pd.concat([bundle.reference, candidates], ignore_index=True, sort=False)
    reference = combined.index[: len(bundle.reference)]
    query = combined.index[len(bundle.reference) :]
    anchor = anchor_features(combined, reference, query)
    anchor.index = candidates.index
    features = feature_matrix(static, anchor, candidates.index, bundle.specification["feature_set"])
    features = features.reindex(columns=bundle.feature_columns, fill_value=0)
    probability = positive_probability(bundle.model, features)
    prediction = decode(candidates, probability, bundle.decoder["threshold"], bundle.decoder["rank_cap"], bundle.decoder["union_baseline"])
    result = candidates[KEYS + [column for column in ["community_name", "school_name"] if column in candidates]].copy()
    result["probability"] = probability
    result["prediction"] = prediction
    return result
