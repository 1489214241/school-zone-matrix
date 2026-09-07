import numpy as np
import pandas as pd

from school_zone_matrix.features import anchor_features


def test_query_label_mutation_does_not_change_anchor_features():
    rows = []
    for community, x, group in [("c1", 0.0, "g1"), ("c2", 100.0, "g2"), ("c3", 200.0, "g3")]:
        for school in ["s1", "s2"]:
            rows.append({"community_id": community, "school_id": school, "group_id": group, "x": x, "y": 0.0, "label": int((community != "c2") == (school == "s1")), "observed": True})
    data = pd.DataFrame(rows)
    reference = data.index[data.community_id.isin(["c1", "c2"])]
    query = data.index[data.community_id.eq("c3")]
    original = data.copy()
    tampered = data.copy()
    tampered.loc[query, "label"] = 1 - tampered.loc[query, "label"]
    for frame in [original, tampered]:
        frame.loc[query, ["label", "observed"]] = [np.nan, False]
    pd.testing.assert_frame_equal(anchor_features(original, reference, query), anchor_features(tampered, reference, query))
