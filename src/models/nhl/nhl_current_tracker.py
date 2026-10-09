from pathlib import Path

import pandas as pd

FEATURES_FILE = Path(
    "data/processed/nhl_features.csv"
)

PREDICTIONS_FILE = Path(
    "data/processed/nhl_live_predictions.csv"
)


def main():
    if not PREDICTIONS_FILE.exists():
        print(
            "No live predictions found."
        )

        return

    predictions = pd.read_csv(
        PREDICTIONS_FILE
    )

    games = pd.read_csv(
        FEATURES_FILE
    )

    if predictions.empty:
        print(
            "No live predictions found."
        )

        return

    predictions[
        "game_id"
    ] = predictions[
        "game_id"
    ].astype(str)

    games[
        "game_id"
    ] = games[
        "game_id"
    ].astype(str)

    completed = games[
        [
            "game_id",
            "away_team",
            "home_team",
            "away_score",
            "home_score",
            "home_win"
        ]
    ].copy()

    tracked = predictions.merge(
        completed,
        on="game_id",
        how="inner",
        suffixes=(
            "_logged",
            "_result"
        )
    )

    if tracked.empty:
        print(
            "No logged predictions have "
            "completed games yet."
        )

        return

    tracked[
        "actual_winner"
    ] = tracked.apply(
        lambda row:
        row["home_team_result"]
        if row["home_win"] == 1
        else row["away_team_result"],
        axis=1
    )

    tracked[
        "correct"
    ] = (
        tracked["predicted_winner"]
        == tracked["actual_winner"]
    )

    tracked = tracked.sort_values(
        [
            "game_date",
            "game_id"
        ]
    ).reset_index(
        drop=True
    )

    print()
    print(
        "NHL Live Prediction Tracker"
    )

    print(
        "==========================="
    )

    correct = 0

    for index, row in tracked.iterrows():
        if row["correct"]:
            correct += 1

        total = index + 1

        accuracy = (
            correct / total
        )

        predicted_probability = (
            row["home_probability"]
            if row["predicted_winner"]
            == row["home_team_logged"]
            else row["away_probability"]
        ) * 100

        result = (
            "CORRECT"
            if row["correct"]
            else "WRONG"
        )

        print()
        print(
            f"{row['away_team_logged']} "
            f"@ {row['home_team_logged']}"
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
        tracked
    )

    total_correct = int(
        tracked["correct"].sum()
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