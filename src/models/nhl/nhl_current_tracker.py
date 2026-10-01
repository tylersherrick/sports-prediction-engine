from pathlib import Path

import joblib
import pandas as pd

FEATURES_FILE = Path(
    "data/processed/nhl_features.csv"
)

MODEL_FILE = Path(
    "models/nhl_winner.joblib"
)

CURRENT_SEASON = 2027


def main():
    data = pd.read_csv(
        FEATURES_FILE
    )

    saved = joblib.load(
        MODEL_FILE
    )

    model = saved[
        "model"
    ]

    features = saved[
        "features"
    ]

    current = data[
        data["season"] == CURRENT_SEASON
    ].copy()

    current = current.sort_values(
        [
            "date",
            "game_id"
        ]
    ).reset_index(
        drop=True
    )

    if current.empty:
        print(
            "No completed 2027 NHL games found."
        )

        return

    probabilities = model.predict_proba(
        current[features]
    )[:, 1]

    current[
        "home_win_probability"
    ] = probabilities

    current[
        "predicted_home_win"
    ] = (
        probabilities >= 0.5
    ).astype(int)

    current[
        "predicted_winner"
    ] = current.apply(
        lambda row:
        row["home_team"]
        if row["predicted_home_win"] == 1
        else row["away_team"],
        axis=1
    )

    current[
        "actual_winner"
    ] = current.apply(
        lambda row:
        row["home_team"]
        if row["home_win"] == 1
        else row["away_team"],
        axis=1
    )

    current[
        "correct"
    ] = (
        current["predicted_home_win"]
        == current["home_win"]
    )

    print()
    print(
        "NHL 2026-27 Current Season Tracker"
    )

    print(
        "=================================="
    )

    correct = 0

    for index, row in current.iterrows():
        if row["correct"]:
            correct += 1

        total = index + 1

        accuracy = (
            correct / total
        )

        home_probability = (
            row["home_win_probability"]
            * 100
        )

        away_probability = (
            100
            - home_probability
        )

        predicted_probability = (
            home_probability
            if row["predicted_home_win"] == 1
            else away_probability
        )

        result = (
            "CORRECT"
            if row["correct"]
            else "WRONG"
        )

        print()
        print(
            f"{row['away_team']} "
            f"@ {row['home_team']}"
        )

        print(
            f"Final: "
            f"{int(row['away_score'])}-"
            f"{int(row['home_score'])}"
        )

        print(
            f"Prediction: "
            f"{row['predicted_winner']} "
            f"({predicted_probability:.1f}%)"
        )

        print(
            f"Actual: "
            f"{row['actual_winner']}"
        )

        print(
            f"Result: {result}"
        )

        print(
            f"Running record: "
            f"{correct}-{total - correct} "
            f"({accuracy:.1%})"
        )

    total_games = len(
        current
    )

    total_correct = int(
        current["correct"].sum()
    )

    total_accuracy = (
        total_correct
        / total_games
    )

    print()
    print(
        "OVERALL"
    )

    print(
        "======="
    )

    print(
        f"Games: {total_games}"
    )

    print(
        f"Record: "
        f"{total_correct}-"
        f"{total_games - total_correct}"
    )

    print(
        f"Accuracy: "
        f"{total_accuracy:.1%}"
    )


if __name__ == "__main__":
    main()