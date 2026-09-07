import pandas as pd

from school_zone_matrix.data import prepare_candidates


def test_prepare_creates_complete_candidate_universe():
    communities = pd.DataFrame(
        [
            {"community_id": "c1", "community_name": "星河一期", "x": 0.0, "y": 0.0},
            {"community_id": "c2", "community_name": "月湖花园", "x": 1000.0, "y": 0.0},
        ]
    )
    schools = pd.DataFrame([{"school_id": "s1", "school_name": "一小"}, {"school_id": "s2", "school_name": "二小"}])
    descriptions = pd.DataFrame([{"school_id": "s1", "description": "范围包括星河一期"}, {"school_id": "s2", "description": "月湖路以东"}])
    result = prepare_candidates(communities, schools, descriptions)
    assert len(result) == 4
    assert not result.duplicated(["community_id", "school_id"]).any()
    assert result.query("community_id == 'c1' and school_id == 's1'").direct_name_match.item() == 1
