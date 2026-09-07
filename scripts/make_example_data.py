from pathlib import Path

import pandas as pd


def main():
    output = Path(__file__).resolve().parents[1] / "examples" / "synthetic"
    output.mkdir(parents=True, exist_ok=True)
    schools = pd.DataFrame(
        [
            {"school_id": "school-east", "school_name": "东区实验小学"},
            {"school_id": "school-west", "school_name": "西区实验小学"},
        ]
    )
    descriptions = pd.DataFrame(
        [
            {"school_id": "school-east", "description": "东区道路围合范围，含东苑一期。"},
            {"school_id": "school-west", "description": "西区道路围合范围，含西苑一期。"},
        ]
    )
    communities = []
    for index in range(30):
        east = index % 2 == 0
        communities.append(
            {
                "community_id": f"community-{index:02d}",
                "community_name": ("东苑" if east else "西苑") + f"{index + 1}期",
                "x": (10_000 if east else 20_000) + index * 8,
                "y": 30_000 + index * 12,
                "group_id": f"group-{index:02d}",
                "outer_fold": index % 5,
            }
        )
    communities = pd.DataFrame(communities)
    new_communities = pd.DataFrame(
        [
            {"community_id": "new-east", "community_name": "东苑新城", "x": 10_090, "y": 30_090, "group_id": "new-east", "outer_fold": 0},
            {"community_id": "new-west", "community_name": "西苑新城", "x": 20_090, "y": 30_090, "group_id": "new-west", "outer_fold": 0},
        ]
    )
    pairs = communities.assign(_key=1).merge(schools.assign(_key=1), on="_key").drop(columns="_key")
    pairs["label"] = ((pairs.community_id.str[-2:].astype(int) % 2 == 0) & pairs.school_id.eq("school-east")) | ((pairs.community_id.str[-2:].astype(int) % 2 == 1) & pairs.school_id.eq("school-west"))
    labels = pairs[["community_id", "school_id"]].copy()
    labels["label"] = pairs.label.astype(int)
    labels["observed"] = True
    communities.to_csv(output / "communities.csv", index=False)
    new_communities.to_csv(output / "new_communities.csv", index=False)
    schools.to_csv(output / "schools.csv", index=False)
    descriptions.to_csv(output / "descriptions.csv", index=False)
    labels.to_csv(output / "labels.csv", index=False)
    print(f"wrote synthetic example to {output}")


if __name__ == "__main__":
    main()
