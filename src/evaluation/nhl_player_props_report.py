import joblib
import numpy as np
import pandas as pd


SKATER_FILE = "data/processed/nhl_skater_features.csv"

TOP_PICKS_PER_TEAM = 3

MODELS = {
    "Shots on Goal": {
        "path": "models/nhl_sog.joblib",
        "target": "shots_on_goal"
    },
    "Points": {
        "path": "models/nhl_points.joblib",
        "target": "points"
    },
    "Goals": {
        "path": "models/nhl_goals.joblib",
        "target": "goals"
    },
    "Assists": {
        "path": "models/nhl_assists.joblib",
        "target": "assists"
    }
}


def load_model(path):
    saved = joblib.load(path)

    return (
        saved["model"],
        saved["features"]
    )


def find_reference_feature(
    features,
    target
):
    preferred = [
        f"{target}_last_10",
        f"{target}_last_5",
        f"{target}_season"
    ]

    for feature in preferred:
        if feature in features:
            return feature

    return None


def half_point_line(value):
    if pd.isna(value):
        return np.nan

    return (
        np.floor(
            max(
                float(value),
                0
            )
        )
        + 0.5
    )


def prepare_betting_picks(
    test,
    predicted,
    features,
    target
):
    reference_feature = find_reference_feature(
        features,
        target
    )

    if reference_feature is None:
        raise ValueError(
            f"Could not find rolling reference "
            f"feature for {target}"
        )

    picks = test.copy()

    picks["prediction"] = predicted

    picks["prop_line"] = picks[
        reference_feature
    ].apply(
        half_point_line
    )

    picks = picks.dropna(
        subset=[
            "prop_line",
            "prediction",
            target
        ]
    ).copy()

    picks["edge"] = (
        picks["prediction"]
        - picks["prop_line"]
    )

    picks["abs_edge"] = (
        picks["edge"].abs()
    )

    picks["pick"] = np.where(
        picks["edge"] >= 0,
        "OVER",
        "UNDER"
    )

    picks["hit"] = np.where(
        picks["pick"] == "OVER",
        picks[target] > picks["prop_line"],
        picks[target] < picks["prop_line"]
    )

    picks = picks.sort_values(
        [
            "game_id",
            "team_id",
            "abs_edge"
        ],
        ascending=[
            True,
            True,
            False
        ]
    )

    picks = (
        picks
        .groupby(
            [
                "game_id",
                "team_id"
            ],
            group_keys=False
        )
        .head(
            TOP_PICKS_PER_TEAM
        )
        .copy()
    )

    return (
        picks,
        reference_feature
    )


def edge_bucket_report(picks):
    buckets = [
        (
            "0.00-0.24",
            0.00,
            0.25
        ),
        (
            "0.25-0.49",
            0.25,
            0.50
        ),
        (
            "0.50-0.74",
            0.50,
            0.75
        ),
        (
            "0.75-0.99",
            0.75,
            1.00
        ),
        (
            "1.00+",
            1.00,
            float("inf")
        )
    ]

    results = []

    for (
        label,
        minimum,
        maximum
    ) in buckets:
        bucket = picks[
            (
                picks["abs_edge"] >= minimum
            )
            & (
                picks["abs_edge"] < maximum
            )
        ]

        if bucket.empty:
            continue

        results.append({
            "label": label,
            "picks": len(
                bucket
            ),
            "hits": int(
                bucket["hit"].sum()
            ),
            "accuracy": bucket[
                "hit"
            ].mean()
        })

    return results


def evaluate_model(
    name,
    config,
    skaters
):
    model, features = load_model(
        config["path"]
    )

    target = config["target"]

    required = (
        features
        + [
            target,
            "season",
            "game_id",
            "team_id"
        ]
    )

    data = skaters.dropna(
        subset=[
            column
            for column in required
            if column in skaters.columns
        ]
    ).copy()

    current_season = data[
        "season"
    ].max()

    test = data[
        data["season"] == current_season
    ].copy()

    if test.empty:
        return None

    predicted = model.predict(
        test[
            features
        ]
    )

    picks, reference_feature = (
        prepare_betting_picks(
            test,
            predicted,
            features,
            target
        )
    )

    overs = picks[
        picks["pick"] == "OVER"
    ]

    unders = picks[
        picks["pick"] == "UNDER"
    ]

    hits = int(
        picks["hit"].sum()
    )

    return {
        "name": name,
        "season": int(
            current_season
        ),
        "reference": reference_feature,
        "picks": len(
            picks
        ),
        "hits": hits,
        "losses": (
            len(picks)
            - hits
        ),
        "accuracy": picks[
            "hit"
        ].mean(),
        "average_edge": picks[
            "abs_edge"
        ].mean(),
        "over_picks": len(
            overs
        ),
        "over_hits": int(
            overs["hit"].sum()
        ),
        "over_accuracy": (
            overs["hit"].mean()
            if not overs.empty
            else None
        ),
        "under_picks": len(
            unders
        ),
        "under_hits": int(
            unders["hit"].sum()
        ),
        "under_accuracy": (
            unders["hit"].mean()
            if not unders.empty
            else None
        ),
        "buckets": edge_bucket_report(
            picks
        )
    }


def print_report(result):
    print(result["name"])
    print("-" * 72)

    print(
        f"Reference Feature:  "
        f"{result['reference']}"
    )

    print(
        f"Top Picks:          "
        f"{result['picks']:,}"
    )

    print(
        f"Record:             "
        f"{result['hits']}-"
        f"{result['losses']}"
    )

    print(
        f"Hit Rate:           "
        f"{result['accuracy']:.1%}"
    )

    print(
        f"Average Edge:       "
        f"{result['average_edge']:.2f}"
    )

    if result["over_accuracy"] is not None:
        print(
            f"Overs:              "
            f"{result['over_hits']}-"
            f"{result['over_picks'] - result['over_hits']} | "
            f"{result['over_accuracy']:.1%}"
        )

    if result["under_accuracy"] is not None:
        print(
            f"Unders:             "
            f"{result['under_hits']}-"
            f"{result['under_picks'] - result['under_hits']} | "
            f"{result['under_accuracy']:.1%}"
        )

    print()
    print("By Model Edge:")

    for bucket in result[
        "buckets"
    ]:
        print(
            f"  {bucket['label']:<10} "
            f"{bucket['hits']}-"
            f"{bucket['picks'] - bucket['hits']} | "
            f"{bucket['accuracy']:.1%} | "
            f"{bucket['picks']} picks"
        )

    print()


def main():
    skaters = pd.read_csv(
        SKATER_FILE,
        dtype={
            "game_id": str,
            "player_id": str,
            "team_id": str
        }
    )

    results = []

    for name, config in MODELS.items():
        result = evaluate_model(
            name,
            config,
            skaters
        )

        if result is not None:
            results.append(
                result
            )

    print()
    print(
        "NHL SKATER PROP BETTING REPORT"
    )
    print("=" * 72)

    if results:
        print(
            f"Evaluation Season: "
            f"{results[0]['season']}"
        )

    print(
        f"Selection: Top "
        f"{TOP_PICKS_PER_TEAM} model edges "
        f"per team per game"
    )

    print()

    for result in results:
        print_report(
            result
        )

    print("=" * 72)


if __name__ == "__main__":
    main()