from pathlib import Path
from uuid import uuid4

import pandas as pd

from school_zone_matrix.audit import audit_gold_observability


def test_audit_flags_small_polygon_and_building_granularity_without_editing_labels():
    schools = pd.DataFrame([
        {"school_id": "normal", "school_name": "正常学校"},
        {"school_id": "campus", "school_name": "校园小框学校"},
        {"school_id": "building", "school_name": "按栋划分学校"},
    ])
    descriptions = pd.DataFrame([
        {"school_id": "normal", "description": "星河花园"},
        {"school_id": "campus", "description": "甲花园、乙花园"},
        {"school_id": "building", "description": "益田村1-43栋"},
    ])
    rows = []
    for community in ("c1", "c2"):
        for school in schools.school_id:
            rows.append({
                "community_id": community,
                "school_id": school,
                "label": int(community == "c1" and school == "normal"),
                "observed": True,
            })
    labels = pd.DataFrame(rows)
    before = labels.copy(deep=True)
    geometry = pd.DataFrame([
        {"school_id": "normal", "polygon_area_m2": 300_000},
        {"school_id": "campus", "polygon_area_m2": 1_000},
        {"school_id": "building", "polygon_area_m2": 200_000},
    ])
    output = Path("test_artifacts") / f"audit-{uuid4().hex}"
    result, report = audit_gold_observability(labels, schools, descriptions, output, geometry)

    classes = result.set_index("school_id").audit_class.to_dict()
    assert classes == {
        "normal": "has_observed_positive",
        "campus": "small_polygon_zero_positive_review",
        "building": "building_rule_centroid_mismatch_review",
    }
    assert report["review_school_count"] == 2
    assert report["labels_modified"] is False
    pd.testing.assert_frame_equal(labels, before)
    assert (output / "school_observability_audit.csv").exists()
    assert (output / "gold_observability_report.json").exists()
