from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

INPUT_FILE = Path(
    "data/processed/nhl_features.csv"
)

MODEL_FILE = Path(
    "models/nhl_winner.joblib"
)

TRAIN_SEASONS = [
    2016,
    2017,
    2018,
    2019,
    2020,
    2021,
    2022,
    2023,
    2024,
    2025,
]

ROLLING_WINDOWS = [
    5,
    10,
    15
]

ROLLING_STATS = [
    "goals_for",
    "goals_against",
    "goal_diff",
    "shots_for",
    "shots_against",
    "shot_diff",
    "shooting_pct",
    "save_pct",
    "power_play_pct",
    "penalty_minutes",
    "win"
]

SEASON_STATS = [
    "goals_for",
    "goals_against",
    "goal_diff",
    "shots_for",
    "shots_against",
    "shot_diff",
    "shooting_pct",
    "save_pct",
    "power_play_pct",
    "win"
]

VENUE_STATS = [
    "goals_for",
    "goals_against",
    "goal_diff",
    "shots_for",
    "shots_against",
    "shot_diff",
    "shooting_pct",
    "save_pct",
    "power_play_pct",
    "win"
]

CONTEXT_FEATURES = [
    "home_days_rest",
    "away_days_rest",
    "home_back_to_back",
    "away_back_to_back",
    "rest_diff",
    "games_played_diff",
    "home_games_played"
]

ROAD_FEATURES = [
    "home_road_streak",
    "away_road_streak",
    "road_streak_diff",
    "home_home_after_road_trip",
    "away_home_after_road_trip",
    "home_after_road_trip_diff"
]


def build_features():
    features = []

    for stat in SEASON_STATS:
        features.extend([
            f"home_{stat}_season",
            f"away_{stat}_season",
            f"{stat}_season_diff"
        ])

    for stat in VENUE_STATS:
        features.extend([
            f"home_{stat}_venue_season",
            f"away_{stat}_venue_season",
            f"{stat}_venue_season_diff"
        ])

    for window in ROLLING_WINDOWS:
        for stat in ROLLING_STATS:
            features.extend([
                f"home_{stat}_last_{window}",
                f"away_{stat}_last_{window}",
                f"{stat}_last_{window}_diff"
            ])

    features.extend(
        CONTEXT_FEATURES
    )

    features.extend(
        ROAD_FEATURES
    )

    return features


def main():
    data = pd.read_csv(
        INPUT_FILE
    )

    features = build_features()

    train = data[
        data["season"].isin(
            TRAIN_SEASONS
        )
    ].copy()

    X_train = train[
        features
    ]

    y_train = train[
        "home_win"
    ]

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=4,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "features": features,
            "train_seasons": TRAIN_SEASONS
        },
        MODEL_FILE
    )

    print("NHL Winner Production Model")
    print("===========================")
    print(
        f"Training seasons: "
        f"{TRAIN_SEASONS}"
    )
    print(
        f"Training games: "
        f"{len(train)}"
    )
    print(
        f"Features: "
        f"{len(features)}"
    )
    print()
    print(
        f"Saved: {MODEL_FILE}"
    )


if __name__ == "__main__":
    main()