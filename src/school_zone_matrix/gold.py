from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


KEYS = ["community_id", "school_id"]


def _observed(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    normalized = values.fillna(False).astype(str).str.strip().str.lower()
    valid = normalized.isin({"true", "false", "1", "0"})
    if not valid.all():
        raise ValueError("observed must contain only true/false or 1/0")
    return normalized.isin({"true", "1"})


def build_hybrid_gold(
    base_labels: pd.DataFrame,
    official_positive_edges: pd.DataFrame,
    output_dir: str | Path,
    resolved_negative_edges: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build complete binary Gold rows with official mentions as hard positives.

    ``base_labels`` supplies the complete community × school universe and any
    quality-controlled spatial positives. ``official_positive_edges`` contains
    already resolved official residence mentions.  An official edge is always
    positive, even when the base label is 0 or unobserved.

    ``resolved_negative_edges`` is optional and must contain separately
    reviewed zero relations. It can resolve an unobserved base edge to zero,
    but cannot conflict with an official positive or a base positive.

    Communities with no positive evidence or any residual unknown relation are
    excluded as whole rows from the complete benchmark. Unknowns are never
    silently converted to zero.
    """
    required_base = set(KEYS + ["label", "observed"])
    required_official = set(KEYS)
    if missing := required_base - set(base_labels):
        raise ValueError(f"base_labels missing columns: {sorted(missing)}")
    if missing := required_official - set(official_positive_edges):
        raise ValueError(f"official_positive_edges missing columns: {sorted(missing)}")
    if base_labels.duplicated(KEYS).any():
        raise ValueError("base_labels contains duplicate relation keys")

    base = base_labels.copy()
    base["community_id"] = base.community_id.astype(str)
    base["school_id"] = base.school_id.astype(str)
    base["observed"] = _observed(base.observed)
    base["label"] = pd.to_numeric(base.label, errors="coerce")
    if not base.loc[base.observed, "label"].isin([0, 1]).all():
        raise ValueError("observed base labels must be binary")

    official = official_positive_edges.copy()
    official["community_id"] = official.community_id.astype(str)
    official["school_id"] = official.school_id.astype(str)
    base_keys = pd.MultiIndex.from_frame(base[KEYS])
    official_keys = pd.MultiIndex.from_frame(official[KEYS].drop_duplicates())
    unknown = official_keys.difference(base_keys)
    if len(unknown):
        raise ValueError(f"official_positive_edges contains {len(unknown)} keys outside the base universe")

    if resolved_negative_edges is None:
        negative = pd.DataFrame(columns=KEYS + ["evidence"])
    else:
        negative = resolved_negative_edges.copy()
        if missing := set(KEYS) - set(negative):
            raise ValueError(f"resolved_negative_edges missing columns: {sorted(missing)}")
        negative["community_id"] = negative.community_id.astype(str)
        negative["school_id"] = negative.school_id.astype(str)
        if negative.duplicated(KEYS).any():
            raise ValueError("resolved_negative_edges contains duplicate relation keys")
        if "label" in negative and not pd.to_numeric(negative.label, errors="coerce").eq(0).all():
            raise ValueError("resolved_negative_edges label column must contain only 0")
        if "evidence" not in negative:
            negative["evidence"] = "reviewed_official_text_negative"
        negative["evidence"] = negative.evidence.fillna("reviewed_official_text_negative").astype(str)

    negative_keys = pd.MultiIndex.from_frame(negative[KEYS])
    unknown_negative = negative_keys.difference(base_keys)
    if len(unknown_negative):
        raise ValueError(f"resolved_negative_edges contains {len(unknown_negative)} keys outside the base universe")
    positive_negative_overlap = negative_keys.intersection(official_keys)
    if len(positive_negative_overlap):
        raise ValueError("resolved_negative_edges conflicts with official_positive_edges")
    base_index = pd.MultiIndex.from_frame(base[KEYS])
    negative_mask = base_index.isin(negative_keys)
    if (base.loc[negative_mask, "observed"] & base.loc[negative_mask, "label"].eq(1)).any():
        raise ValueError("resolved_negative_edges conflicts with an observed base positive")
    negative_overrides = base.loc[negative_mask, KEYS + ["label", "observed"]].copy()
    negative_overrides = negative_overrides.merge(
        negative[KEYS + ["evidence"]], on=KEYS, how="left", validate="one_to_one"
    )
    negative_overrides["old_state"] = [
        str(int(label)) if observed else "U"
        for label, observed in zip(negative_overrides.label.fillna(0), negative_overrides.observed)
    ]
    negative_overrides["new_label"] = 0
    base.loc[negative_mask, "label"] = 0
    base.loc[negative_mask, "observed"] = True

    spatial = base.loc[base.observed & base.label.eq(1), KEYS].copy()
    spatial["evidence"] = "base_observed_positive"
    if "evidence" not in official:
        official["evidence"] = "official_residence_field"
    official["evidence"] = official.evidence.fillna("official_residence_field").astype(str)
    evidence = pd.concat(
        [spatial, official[KEYS + ["evidence"]]], ignore_index=True
    ).drop_duplicates()
    positive = (
        evidence.groupby(KEYS).evidence
        .agg(lambda values: "|".join(sorted(set(values))))
        .rename("positive_evidence")
        .reset_index()
    )

    work = base.merge(positive, on=KEYS, how="left", validate="one_to_one")
    unresolved = ~work.observed & work.positive_evidence.isna()
    work["label"] = work.positive_evidence.notna().astype(int)
    work["observed"] = True
    positive_community_ids = set(work.loc[work.label.eq(1), "community_id"])
    unresolved_community_ids = set(work.loc[unresolved, "community_id"])
    included_ids = positive_community_ids - unresolved_community_ids
    complete = work.loc[work.community_id.isin(included_ids), KEYS + ["label", "observed"]]
    complete = complete.sort_values(KEYS).reset_index(drop=True)

    all_community_ids = set(base.community_id)
    excluded_ids = sorted(all_community_ids - included_ids)
    old_state = base[KEYS].copy()
    old_state["old_state"] = [
        str(int(label)) if observed else "U"
        for label, observed in zip(base.label.fillna(0), base.observed)
    ]
    overrides = official[KEYS + ["evidence"]].drop_duplicates(KEYS).merge(
        old_state, on=KEYS, how="left", validate="one_to_one"
    )
    overrides = overrides.loc[~overrides.old_state.eq("1")].copy()
    overrides["new_label"] = 1
    overrides["included_in_complete_binary_gold"] = overrides.community_id.isin(included_ids)

    complete_positive = positive.loc[positive.community_id.isin(included_ids)].copy()
    school_positive = complete_positive.groupby("school_id").size()
    school_ids = sorted(set(base.school_id))
    zero_positive_schools = [school_id for school_id in school_ids if school_id not in school_positive]
    report = {
        "school_count": len(school_ids),
        "source_community_count": len(all_community_ids),
        "base_positive_edges": len(spatial),
        "official_positive_edges": len(official.drop_duplicates(KEYS)),
        "hybrid_positive_edges_all_communities": len(positive),
        "hybrid_positive_edges_in_complete_binary_gold": len(complete_positive),
        "official_overrides": len(overrides),
        "resolved_negative_edges": len(negative),
        "resolved_negative_unknowns_to_zero": int(negative_overrides.old_state.eq("U").sum()),
        "included_complete_binary_communities": len(included_ids),
        "excluded_communities": len(excluded_ids),
        "excluded_no_positive_communities": len(all_community_ids - positive_community_ids),
        "excluded_residual_unobserved_communities": len(positive_community_ids & unresolved_community_ids),
        "residual_unobserved_edges_not_converted_to_zero": int(unresolved.sum()),
        "complete_binary_edges": len(complete),
        "complete_binary_has_unknown": False,
        "zero_positive_school_ids": zero_positive_schools,
        "policy": "official residence positives are hard positives; reviewed negatives resolve only non-positive edges",
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    complete.to_csv(output / "hybrid_complete_binary_labels.csv", index=False)
    positive.to_csv(output / "hybrid_all_positive_edges_audit.csv", index=False)
    complete_positive.to_csv(output / "hybrid_positive_edges.csv", index=False)
    overrides.sort_values(KEYS).to_csv(output / "official_positive_overrides.csv", index=False)
    negative_overrides.sort_values(KEYS).to_csv(output / "resolved_negative_overrides.csv", index=False)
    pd.DataFrame({
        "community_id": excluded_ids,
        "reason": [
            "no_positive_evidence_excluded_from_complete_binary_gold"
            if community_id not in positive_community_ids
            else "residual_unobserved_relation_not_converted_to_zero"
            for community_id in excluded_ids
        ],
    }).to_csv(output / "excluded_communities.csv", index=False)
    (output / "hybrid_gold_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return complete, report
