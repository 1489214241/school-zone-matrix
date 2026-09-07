from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

KEYS = ["community_id", "school_id"]


def normalize_name(value: object) -> str:
    value = unicodedata.normalize("NFKC", str(value)).lower()
    return re.sub(r"[\s·•（）()\-—_]", "", value)


def family_name(value: object) -> str:
    value = normalize_name(value)
    for pattern in [
        r"(?:第?[一二三四五六七八九十百零0-9]+期)$",
        r"(?:[一二三四五六七八九十百零0-9]+区)$",
        r"(?:东区|西区|南区|北区)$",
    ]:
        value = re.sub(pattern, "", value)
    return value


def hash_bucket(value: object, buckets: int, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % buckets


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def ensure_xy(communities: pd.DataFrame, projected_crs: str = "EPSG:32650") -> pd.DataFrame:
    communities = communities.copy()
    if {"x", "y"} <= set(communities.columns):
        communities[["x", "y"]] = communities[["x", "y"]].apply(pd.to_numeric, errors="raise")
        return communities
    require_columns(communities, ["lon", "lat"], "communities")
    transformer = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    communities["x"], communities["y"] = transformer.transform(
        pd.to_numeric(communities.lon, errors="raise").to_numpy(),
        pd.to_numeric(communities.lat, errors="raise").to_numpy(),
    )
    return communities


def spatial_groups(communities: pd.DataFrame, near_distance_m: float = 100.0) -> pd.Series:
    """Keep exact-name and near-duplicate communities in the same evaluation group."""
    communities = communities.reset_index(drop=True)
    parent = list(range(len(communities)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    seen: dict[str, int] = {}
    for index, name in enumerate(communities.community_name.map(normalize_name)):
        if name in seen:
            union(index, seen[name])
        else:
            seen[name] = index
    if len(communities) > 1:
        points = communities[["x", "y"]].to_numpy(float)
        for left, right in cKDTree(points).query_pairs(near_distance_m):
            union(left, right)
    return pd.Series(
        [str(communities.loc[find(index), "community_id"]) for index in range(len(communities))],
        index=communities.index,
        name="group_id",
    )


def prepare_candidates(
    communities: pd.DataFrame,
    schools: pd.DataFrame,
    descriptions: pd.DataFrame,
    projected_crs: str = "EPSG:32650",
    folds: int = 5,
    split_salt: str = "school-zone-matrix-v1",
) -> pd.DataFrame:
    require_columns(communities, ["community_id", "community_name"], "communities")
    require_columns(schools, ["school_id", "school_name"], "schools")
    require_columns(descriptions, ["school_id", "description"], "descriptions")
    if communities.community_id.duplicated().any() or schools.school_id.duplicated().any():
        raise ValueError("community_id and school_id must be unique")
    communities = ensure_xy(communities, projected_crs)
    if "group_id" not in communities:
        communities["group_id"] = spatial_groups(communities).to_numpy()
    if "outer_fold" not in communities:
        communities["outer_fold"] = communities.group_id.map(lambda value: hash_bucket(value, folds, split_salt))
    descriptions = descriptions.groupby("school_id", as_index=False).description.agg("\n".join)
    schools = schools.merge(descriptions, on="school_id", how="left", validate="one_to_one")
    schools["description"] = schools.description.fillna("")
    communities = communities.copy()
    communities["family_name"] = communities.community_name.map(family_name)
    pairs = communities.assign(_key=1).merge(schools.assign(_key=1), on="_key", validate="many_to_many").drop(columns="_key")
    normalized_description = pairs.description.map(normalize_name)
    normalized_community = pairs.community_name.map(normalize_name)
    pairs["direct_name_match"] = [int(bool(name) and name in text) for name, text in zip(normalized_community, normalized_description)]
    pairs["description_char_count"] = pairs.description.str.len().astype(float)
    pairs["description_has_boundary"] = pairs.description.str.contains(r"以东|以西|以南|以北|道路|路段|交汇|围合", regex=True).astype(int)
    pairs["description_has_admin"] = pairs.description.str.contains(r"社区|居委会|工作站", regex=True).astype(int)
    pairs["description_has_shared_rule"] = pairs.description.str.contains(r"共享|大学区|可申请|统筹", regex=True).astype(int)
    pairs["baseline"] = pairs.direct_name_match.astype(int)
    if len(pairs) != len(communities) * len(schools) or pairs.duplicated(KEYS).any():
        raise AssertionError("candidate universe is not a complete unique Cartesian product")
    return pairs


def merge_extra_features(candidates: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    require_columns(extra, KEYS, "extra_features")
    if extra.duplicated(KEYS).any():
        raise ValueError("extra_features contains duplicate relation keys")
    overlap = (set(extra.columns) & set(candidates.columns)) - set(KEYS)
    if overlap:
        raise ValueError(f"extra feature names already exist: {sorted(overlap)}")
    forbidden = ("gold", "truth", "label", "prediction", "probability")
    unsafe = [column for column in extra.columns if column not in KEYS and any(token in column.lower() for token in forbidden)]
    if unsafe:
        raise ValueError(f"forbidden prediction-side feature names: {unsafe}")
    return candidates.merge(extra, on=KEYS, how="left", validate="one_to_one")


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
