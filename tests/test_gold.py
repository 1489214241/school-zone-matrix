from pathlib import Path
from uuid import uuid4

import pandas as pd

from school_zone_matrix.gold import build_hybrid_gold


def test_official_positive_overrides_polygon_and_output_has_no_unknowns():
    base = pd.DataFrame([
        {"community_id": "c1", "school_id": "s1", "label": 0, "observed": True},
        {"community_id": "c1", "school_id": "s2", "label": 1, "observed": True},
        {"community_id": "c2", "school_id": "s1", "label": 0, "observed": True},
        {"community_id": "c2", "school_id": "s2", "label": 0, "observed": False},
        {"community_id": "c3", "school_id": "s1", "label": 0, "observed": True},
        {"community_id": "c3", "school_id": "s2", "label": 0, "observed": True},
    ])
    official = pd.DataFrame([
        {"community_id": "c1", "school_id": "s1", "evidence": "official_residence_field"},
        {"community_id": "c2", "school_id": "s2", "evidence": "official_residence_field"},
    ])
    output = Path("test_artifacts") / f"hybrid-gold-{uuid4().hex}"

    labels, report = build_hybrid_gold(base, official, output)

    actual = labels.set_index(["community_id", "school_id"]).label.to_dict()
    assert actual == {("c1", "s1"): 1, ("c1", "s2"): 1, ("c2", "s1"): 0, ("c2", "s2"): 1}
    assert labels.observed.all()
    assert set(labels.community_id) == {"c1", "c2"}
    assert report["official_overrides"] == 2
    assert report["excluded_no_positive_communities"] == 1
    assert (output / "hybrid_complete_binary_labels.csv").exists()
    assert (output / "official_positive_overrides.csv").exists()


def test_official_edges_must_belong_to_base_universe():
    base = pd.DataFrame([
        {"community_id": "c1", "school_id": "s1", "label": 0, "observed": True},
    ])
    official = pd.DataFrame([
        {"community_id": "missing", "school_id": "s1"},
    ])
    output = Path("test_artifacts") / f"hybrid-gold-invalid-{uuid4().hex}"

    try:
        build_hybrid_gold(base, official, output)
    except ValueError as error:
        assert "outside the base universe" in str(error)
    else:
        raise AssertionError("unknown official edge was accepted")
