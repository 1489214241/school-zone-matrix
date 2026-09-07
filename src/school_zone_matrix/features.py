from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .data import KEYS, hash_bucket

ANCHOR_COLUMNS = [
    "positive_distance",
    "negative_distance",
    "distance_ratio",
    "positive_2_distance",
    "negative_2_distance",
    "positive_3_distance",
    "negative_3_distance",
    "knn1",
    "knn3",
    "knn5",
    "knn9",
    "nearest_distance",
    "no_training_positive",
    "no_training_negative",
]

RESERVED = {
    *KEYS,
    "community_name",
    "school_name",
    "description",
    "family_name",
    "group_id",
    "outer_fold",
    "label",
    "observed",
    "truth",
    "prediction",
    "probability",
    "x",
    "y",
}


def predictor_columns(candidates: pd.DataFrame) -> list[str]:
    forbidden = ("gold", "truth", "label", "prediction", "probability")
    columns = []
    for column in candidates.columns:
        if column in RESERVED or any(token in column.lower() for token in forbidden):
            continue
        if pd.api.types.is_numeric_dtype(candidates[column]):
            columns.append(column)
    return sorted(columns)


def static_features(
    candidates: pd.DataFrame,
    numeric_columns: list[str],
    school_ids: list[str],
) -> pd.DataFrame:
    output: dict[str, np.ndarray] = {}
    for column in numeric_columns:
        values = pd.to_numeric(candidates.get(column), errors="coerce") if column in candidates else pd.Series(np.nan, index=candidates.index)
        output[f"num_{column}"] = values.fillna(0).to_numpy(float)
        output[f"missing_{column}"] = values.isna().to_numpy(float)
    output["coord_x_km"] = pd.to_numeric(candidates.x, errors="raise").to_numpy(float) / 1000.0
    output["coord_y_km"] = pd.to_numeric(candidates.y, errors="raise").to_numpy(float) / 1000.0
    school_to_index = {school_id: index for index, school_id in enumerate(school_ids)}
    unknown = sorted(set(candidates.school_id) - set(school_to_index))
    if unknown:
        raise ValueError(f"unknown schools at prediction time: {unknown}")
    codes = candidates.school_id.map(school_to_index).to_numpy()
    for school_id, index in school_to_index.items():
        output[f"school_{index:03d}"] = (codes == index).astype(float)
    result = pd.DataFrame(output, index=candidates.index, dtype=np.float32)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("non-finite static feature")
    return result


def _distances(points: np.ndarray, query: np.ndarray) -> np.ndarray:
    result = np.full((len(query), 3), 50_000.0)
    if len(points):
        distance, _ = cKDTree(points).query(query, k=[1, 2, 3])
        result = np.minimum(distance, 50_000.0)
    return result


def anchor_features(data: pd.DataFrame, reference, query) -> pd.DataFrame:
    reference, query = pd.Index(reference), pd.Index(query)
    if set(reference) & set(query):
        raise AssertionError("reference and query rows overlap")
    if set(data.loc[reference, "group_id"]) & set(data.loc[query, "group_id"]):
        raise AssertionError("reference and query spatial groups overlap")
    output = pd.DataFrame(index=query, columns=ANCHOR_COLUMNS, dtype=float)
    ref = data.loc[reference]
    for school_id, q in data.loc[query].groupby("school_id"):
        r = ref[ref.school_id.eq(school_id) & ref.observed & ref.label.notna()]
        locations = q[["x", "y"]].to_numpy(float)
        positive = r[r.label.eq(1)][["x", "y"]].to_numpy(float)
        negative = r[r.label.eq(0)][["x", "y"]].to_numpy(float)
        dp, dn = _distances(positive, locations), _distances(negative, locations)
        votes = []
        if len(r):
            distance, index = cKDTree(r[["x", "y"]].to_numpy(float)).query(locations, k=list(range(1, 10)))
            labels = np.append(r.label.to_numpy(float), 0.0)[index]
            weight = 1.0 / np.maximum(distance, 50.0)
            weight[~np.isfinite(distance)] = 0.0
            for k in [1, 3, 5, 9]:
                denominator = weight[:, :k].sum(axis=1)
                votes.append(np.divide((labels[:, :k] * weight[:, :k]).sum(axis=1), denominator, out=np.zeros(len(q)), where=denominator > 0))
        else:
            votes = [np.zeros(len(q)) for _ in range(4)]
        values = np.column_stack(
            [
                np.log1p(dp[:, 0] / 50),
                np.log1p(dn[:, 0] / 50),
                np.log((dn[:, 0] + 50) / (dp[:, 0] + 50)),
                np.log1p(dp[:, 1] / 50),
                np.log1p(dn[:, 1] / 50),
                np.log1p(dp[:, 2] / 50),
                np.log1p(dn[:, 2] / 50),
                *votes,
                np.log1p(np.minimum(dp[:, 0], dn[:, 0]) / 50),
                np.full(len(q), int(len(positive) == 0)),
                np.full(len(q), int(len(negative) == 0)),
            ]
        )
        output.loc[q.index] = values
    temp = data.loc[query, ["community_id"]].copy()
    temp["positive_distance"] = output.positive_distance
    temp["knn5"] = output.knn5
    output["positive_distance_rank"] = temp.groupby("community_id").positive_distance.rank(method="min")
    output["knn5_rank"] = temp.groupby("community_id").knn5.rank(method="min", ascending=False)
    output["positive_distance_gap"] = temp.positive_distance - temp.groupby("community_id").positive_distance.transform("min")
    output["knn5_gap"] = temp.groupby("community_id").knn5.transform("max") - temp.knn5
    if not np.isfinite(output.to_numpy()).all():
        raise ValueError("non-finite anchor feature")
    return output.astype(np.float32)


def training_anchor_features(data: pd.DataFrame, indices, salt: str) -> pd.DataFrame:
    indices = pd.Index(indices)
    folds = data.loc[indices, "group_id"].map(lambda value: hash_bucket(value, 4, salt))
    chunks = []
    for fold in range(4):
        query = indices[folds.eq(fold)]
        reference = indices[~folds.eq(fold)]
        if len(query):
            chunks.append(anchor_features(data, reference, query))
    return pd.concat(chunks).loc[indices]
