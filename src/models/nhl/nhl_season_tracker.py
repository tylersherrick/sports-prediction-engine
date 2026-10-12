from pathlib import Path

import joblib
import pandas as pd


FEATURES_FILE = Path(
    "data/processed/nhl_features.csv"
)

MODEL_FILE = Path(
    "models/nhl_winner.joblib"
)

OUTPUT_FILE = Path(
    "data/processed/nhl_season_predictions.csv"
)

CURRENT_SEASON = 2027


def main():
    data = pd.read_csv(
        FEATURES_FILE
    )

    saved = joblib.load(
        MODEL_FILE
    )

    model = saved["model"]
    features = saved["features"]

    season = data[
        data["season"]
        == CURRENT_SEASON
    ].copy()

    if season.empty:
        print(
            "No current-season games found."
        )

        return

    probabilities = model.predict_proba(
        season[features]
    )[:, 1]

    season[
        "home_win_probability"
    ] = probabilities

    season[
        "away_win_probability"
    ] = (
        1
        - season["home_win_probability"]
    )

    season[
        "predicted_home_win"
    ] = (
        season["home_win_probability"]
        >= 0.5
    ).astype(int)

    season[
        "predicted_winner"
    ] = season.apply(
        lambda row:
        row["home_team"]
        if row["predicted_home_win"] == 1
        else row["away_team"],
        axis=1
    )

    season[
        "actual_winner"
    ] = season.apply(
        lambda row:
        row["home_team"]
        if row["home_win"] == 1
        else row["away_team"],
        axis=1
    )

    season[
        "prediction_probability"
    ] = season.apply(
        lambda row:
        row["home_win_probability"]
        if row["predicted_home_win"] == 1
        else row["away_win_probability"],
        axis=1
    )

    season[
        "correct"
    ] = (
        season["predicted_home_win"]
        == season["home_win"]
    )

    output_columns = [
        "game_id",
        "season",
        "away_team",
        "home_team",
        "away_score",
        "home_score",
        "away_win_probability",
        "home_win_probability",
        "predicted_winner",
        "prediction_probability",
        "actual_winner",
        "correct"
    ]

    tracker = season[
        output_columns
    ].copy()

    tracker[
        "away_win_probability"
    ] = tracker[
        "away_win_probability"
    ].round(4)

    tracker[
        "home_win_probability"
    ] = tracker[
        "home_win_probability"
    ].round(4)

    tracker[
        "prediction_probability"
    ] = tracker[
        "prediction_probability"
    ].round(4)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    tracker.to_csv(
        OUTPUT_FILE,
        index=False
    )

    correct = int(
        tracker["correct"].sum()
    )

    total = len(
        tracker
    )

    losses = (
        total
        - correct
    )

    accuracy = (
        correct
        / total
    )

    print()
    print(
        "NHL Current Season"
    )

    print(
        "=================="
    )

    print(
        f"Games: {total}"
    )

    print(
        f"Record: "
        f"{correct}-{losses}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.1%}"
    )

    print()
    print(
        f"Saved: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()