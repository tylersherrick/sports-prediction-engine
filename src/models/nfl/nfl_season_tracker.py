from pathlib import Path

import joblib
import pandas as pd


FEATURES_FILE = Path(
    "data/processed/nfl_features.csv"
)

GAMES_FILE = Path(
    "data/processed/nfl_games.csv"
)

MODEL_FILE = Path(
    "models/nfl_2026_predictor.joblib"
)

OUTPUT_FILE = Path(
    "data/processed/nfl_season_predictions.csv"
)

CURRENT_SEASON = 2026


def print_record(
    tracker,
    label
):
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
        correct / total
        if total
        else 0
    )

    print(
        f"{label}: "
        f"{correct}-{losses} "
        f"({accuracy:.1%}) | "
        f"{total} games"
    )


def main():
    data = pd.read_csv(
        FEATURES_FILE
    )

    games = pd.read_csv(
        GAMES_FILE
    )

    saved = joblib.load(
        MODEL_FILE
    )

    model = saved["model"]
    features = saved["features"]

    data["game_id"] = (
        data["game_id"]
        .astype(str)
    )

    games["game_id"] = (
        games["game_id"]
        .astype(str)
    )

    results = games[
        [
            "game_id",
            "away_score",
            "home_score"
        ]
    ].copy()

    data = data.merge(
        results,
        on="game_id",
        how="left"
    )

    data["home_yards_matchup"] = (
        data["home_avg_yards_last_5"]
        - data["away_avg_yards_allowed_last_5"]
    )

    data["away_yards_matchup"] = (
        data["away_avg_yards_last_5"]
        - data["home_avg_yards_allowed_last_5"]
    )

    data["home_points_matchup"] = (
        data["home_avg_points_last_5"]
        - data["away_avg_points_allowed_last_5"]
    )

    data["away_points_matchup"] = (
        data["away_avg_points_last_5"]
        - data["home_avg_points_allowed_last_5"]
    )

    season = data[
        data["season"]
        == CURRENT_SEASON
    ].copy()

    season = season.dropna(
        subset=features + [
            "home_win",
            "away_score",
            "home_score"
        ]
    )

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
    ] = model.predict(
        season[features]
    )

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
        "week",
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

    tracker[
        "away_score"
    ] = tracker[
        "away_score"
    ].astype(int)

    tracker[
        "home_score"
    ] = tracker[
        "home_score"
    ].astype(int)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    tracker.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        "NFL Current Season"
    )

    print(
        "=================="
    )

    print_record(
        tracker,
        "Overall"
    )

    print()
    print(
        "Confidence Records"
    )

    print(
        "=================="
    )

    for threshold in [
        0.60,
        0.65,
        0.70
    ]:
        confident = tracker[
            tracker[
                "prediction_probability"
            ] >= threshold
        ]

        print_record(
            confident,
            f"{threshold:.0%}+"
        )

    print()
    print(
        f"Saved: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()