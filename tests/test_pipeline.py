from pathlib import Path
from uuid import uuid4

import pandas as pd

from school_zone_matrix.data import prepare_candidates
from school_zone_matrix.modeling import crossfit_predict, evaluate_frozen, predict_new, train_release


def test_crossfit_freezes_before_evaluation():
    schools = pd.DataFrame([{"school_id": "east", "school_name": "东校"}, {"school_id": "west", "school_name": "西校"}])
    descriptions = pd.DataFrame([{"school_id": "east", "description": "东片区"}, {"school_id": "west", "description": "西片区"}])
    communities = []
    for index in range(30):
        east = index % 2 == 0
        communities.append({"community_id": f"c{index:02d}", "community_name": f"小区{index}", "x": (0 if east else 10_000) + index, "y": index * 3, "group_id": f"g{index:02d}", "outer_fold": index % 5})
    candidates = prepare_candidates(pd.DataFrame(communities), schools, descriptions)
    labels = candidates[["community_id", "school_id"]].copy()
    labels["label"] = ((candidates.community_id.str[1:].astype(int) % 2 == 0) & candidates.school_id.eq("east")) | ((candidates.community_id.str[1:].astype(int) % 2 == 1) & candidates.school_id.eq("west"))
    labels["label"] = labels.label.astype(int)
    labels["observed"] = True
    run_dir = Path("test_artifacts") / f"pipeline-{uuid4().hex}"
    predictions = crossfit_predict(candidates, labels, run_dir)
    assert predictions.exists()
    assert (run_dir / "all_predictions_frozen.json").exists()
    assert not (run_dir / "evaluation" / "evaluation.json").exists()
    report = evaluate_frozen(predictions, labels, run_dir / "evaluation")
    assert report["overall"]["precision"] >= 0.8
    assert report["overall"]["recall"] >= 0.8


def test_release_model_predicts_unseen_communities():
    schools = pd.DataFrame([{"school_id": "east", "school_name": "东校"}, {"school_id": "west", "school_name": "西校"}])
    descriptions = pd.DataFrame([{"school_id": "east", "description": "东片区"}, {"school_id": "west", "description": "西片区"}])
    communities = []
    for index in range(30):
        east = index % 2 == 0
        communities.append({"community_id": f"c{index:02d}", "community_name": f"小区{index}", "x": (0 if east else 10_000) + index, "y": index * 3, "group_id": f"g{index:02d}"})
    candidates = prepare_candidates(pd.DataFrame(communities), schools, descriptions)
    labels = candidates[["community_id", "school_id"]].copy()
    labels["label"] = ((candidates.community_id.str[1:].astype(int) % 2 == 0) & candidates.school_id.eq("east")) | ((candidates.community_id.str[1:].astype(int) % 2 == 1) & candidates.school_id.eq("west"))
    labels["label"] = labels.label.astype(int)
    labels["observed"] = True
    model_path = Path("test_artifacts") / f"release-{uuid4().hex}.joblib"
    bundle = train_release(candidates, labels, model_path)

    new_communities = pd.DataFrame([
        {"community_id": "new-east", "community_name": "新东苑", "x": 15, "y": 15, "group_id": "new-east"},
        {"community_id": "new-west", "community_name": "新西苑", "x": 10_015, "y": 15, "group_id": "new-west"},
    ])
    result = predict_new(bundle, prepare_candidates(new_communities, schools, descriptions))
    assert len(result) == 4
    assert result.groupby("community_id")["prediction"].sum().ge(1).all()
    assert set(result.columns) >= {"community_id", "school_id", "probability", "prediction"}
