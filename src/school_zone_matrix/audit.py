from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import KEYS, require_columns


BUILDING_RULE = re.compile(
    r"(?:\d+\s*(?:栋|座|幢)|\d+\s*[-—至到]\s*\d+\s*(?:栋|座|幢)|[A-Za-zＡ-Ｚ]\s*[-—至到]\s*[A-Za-zＡ-Ｚ]\s*(?:栋|座|幢)|部队)"
)


def audit_gold_observability(
    labels: pd.DataFrame,
    schools: pd.DataFrame,
    descriptions: pd.DataFrame,
    output_dir: str | Path,
    geometry_stats: pd.DataFrame | None = None,
    small_polygon_threshold_m2: float = 50_000.0,
) -> tuple[pd.DataFrame, dict]:
    """Flag school columns whose current 0/1 Gold is hard to observe.

    This is a diagnostic.  It never edits labels and never converts a flag into
    a positive or negative relation automatically.
    """
    require_columns(labels, KEYS + ["label", "observed"], "labels")
    require_columns(schools, ["school_id", "school_name"], "schools")
    require_columns(descriptions, ["school_id", "description"], "descriptions")
    if labels.duplicated(KEYS).any():
        raise ValueError("labels contains duplicate relation keys")
    if schools.school_id.duplicated().any():
        raise ValueError("schools contains duplicate school_id")
    if geometry_stats is not None:
        require_columns(geometry_stats, ["school_id", "polygon_area_m2"], "geometry_stats")
        if geometry_stats.school_id.duplicated().any():
            raise ValueError("geometry_stats contains duplicate school_id")

    work = labels.copy()
    work["observed"] = work.observed.fillna(False).astype(bool)
    observed_values = pd.to_numeric(work.loc[work.observed, "label"], errors="coerce")
    if observed_values.isna().any() or not observed_values.isin([0, 1]).all():
        raise ValueError("observed labels must be binary")
    work["label_numeric"] = pd.to_numeric(work.label, errors="coerce")

    description = (
        descriptions.assign(description=descriptions.description.fillna("").astype(str))
        .groupby("school_id", as_index=False).description.agg("\n".join)
    )
    result = schools[["school_id", "school_name"]].merge(
        description, on="school_id", how="left", validate="one_to_one"
    )
    result["description"] = result.description.fillna("")
    counts = []
    for school_id in result.school_id:
        rows = work.loc[work.school_id.eq(school_id)]
        counts.append({
            "school_id": school_id,
            "candidate_edges": len(rows),
            "observed_positive": int((rows.observed & rows.label_numeric.eq(1)).sum()),
            "observed_negative": int((rows.observed & rows.label_numeric.eq(0)).sum()),
            "unobserved": int((~rows.observed).sum()),
        })
    result = result.merge(pd.DataFrame(counts), on="school_id", validate="one_to_one")
    if geometry_stats is not None:
        result = result.merge(
            geometry_stats[["school_id", "polygon_area_m2"]],
            on="school_id",
            how="left",
            validate="one_to_one",
        )
    else:
        result["polygon_area_m2"] = pd.NA

    result["has_building_granularity_rule"] = result.description.str.contains(BUILDING_RULE)
    result["small_polygon_candidate"] = (
        pd.to_numeric(result.polygon_area_m2, errors="coerce") < small_polygon_threshold_m2
    )
    classes = []
    for row in result.itertuples():
        if row.observed_positive > 0:
            classes.append("has_observed_positive")
        elif row.observed_negative == 0:
            classes.append("no_observed_labels")
        elif row.small_polygon_candidate:
            classes.append("small_polygon_zero_positive_review")
        elif row.has_building_granularity_rule:
            classes.append("building_rule_centroid_mismatch_review")
        else:
            classes.append("zero_positive_review")
    result["audit_class"] = classes
    result["requires_review"] = result.audit_class.str.endswith("review")
    result = result.sort_values(["requires_review", "audit_class", "school_name"], ascending=[False, True, True])

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "school_observability_audit.csv", index=False, encoding="utf-8-sig")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "GOLD_OBSERVABILITY_DIAGNOSTIC",
        "labels_modified": False,
        "school_count": len(result),
        "review_school_count": int(result.requires_review.sum()),
        "audit_class_counts": dict(Counter(result.audit_class)),
        "small_polygon_threshold_m2": small_polygon_threshold_m2,
        "interpretation": (
            "Flags identify columns requiring source/granularity review. "
            "They do not change 0/1/U labels automatically."
        ),
    }
    (output / "gold_observability_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result, report
