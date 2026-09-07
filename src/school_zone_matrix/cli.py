from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import merge_extra_features, prepare_candidates, read_csv
from .modeling import crossfit_predict, evaluate_frozen, predict_new, train_release


def prepare_command(args):
    candidates = prepare_candidates(read_csv(args.communities), read_csv(args.schools), read_csv(args.descriptions), args.projected_crs)
    if args.extra_features:
        candidates = merge_extra_features(candidates, read_csv(args.extra_features))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output, index=False)
    print(f"wrote {len(candidates)} candidate relations to {args.output}")


def crossfit_command(args):
    path = crossfit_predict(read_csv(args.candidates), read_csv(args.labels), args.output_dir)
    print(f"predictions frozen at {path}")


def evaluate_command(args):
    report = evaluate_frozen(args.predictions, read_csv(args.labels), args.output_dir)
    print(report)


def train_command(args):
    train_release(read_csv(args.candidates), read_csv(args.labels), args.model)
    print(f"release model written to {args.model}")


def predict_command(args):
    bundle = joblib.load(args.model)
    result = predict_new(bundle, read_csv(args.candidates))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "relations.csv", index=False)
    result.pivot(index="school_id", columns="community_id", values="probability").to_csv(output_dir / "probability_matrix.csv")
    result.pivot(index="school_id", columns="community_id", values="prediction").to_csv(output_dir / "relation_matrix.csv")
    print(f"wrote relations and matrices to {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(prog="school-zone-matrix")
    sub = parser.add_subparsers(required=True)
    prepare = sub.add_parser("prepare", help="create the complete school × community candidate universe")
    prepare.add_argument("--communities", required=True)
    prepare.add_argument("--schools", required=True)
    prepare.add_argument("--descriptions", required=True)
    prepare.add_argument("--extra-features")
    prepare.add_argument("--projected-crs", default="EPSG:32650")
    prepare.add_argument("--output", required=True)
    prepare.set_defaults(func=prepare_command)
    crossfit = sub.add_parser("crossfit-predict", help="freeze grouped cross-fitted predictions without scoring Gold")
    crossfit.add_argument("--candidates", required=True)
    crossfit.add_argument("--labels", required=True)
    crossfit.add_argument("--output-dir", required=True)
    crossfit.set_defaults(func=crossfit_command)
    evaluate = sub.add_parser("evaluate", help="score an already frozen prediction with isolated Gold")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--labels", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.set_defaults(func=evaluate_command)
    train = sub.add_parser("train-release", help="train a release bundle from all reliable training labels")
    train.add_argument("--candidates", required=True)
    train.add_argument("--labels", required=True)
    train.add_argument("--model", required=True)
    train.set_defaults(func=train_command)
    predict = sub.add_parser("predict", help="predict new communities and write long/probability/binary matrices")
    predict.add_argument("--model", required=True)
    predict.add_argument("--candidates", required=True)
    predict.add_argument("--output-dir", required=True)
    predict.set_defaults(func=predict_command)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
