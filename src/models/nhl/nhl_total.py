from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


INPUT_FILE = Path(
    "data/processed/nhl_features.csv"
)

MODEL_FILE = Path(
    "models/nhl_total.joblib"
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
    2026
]

FEATURES = [
    "home_goals_for_last_5",
    "away_goals_for_last_5",
    "home_goals_against_last_5",
    "away_goals_against_last_5",

    "home_shots_for_last_5",
    "away_shots_for_last_5",
    "home_shots_against_last_5",
    "away_shots_against_last_5",

    "home_shooting_pct_last_5",
    "away_shooting_pct_last_5",
    "home_save_pct_last_5",
    "away_save_pct_last_5",

    "home_power_play_pct_last_5",
    "away_power_play_pct_last_5",

    "home_goals_for_season",
    "away_goals_for_season",
    "home_goals_against_season",
    "away_goals_against_season",

    "home_shots_for_season",
    "away_shots_for_season",
    "home_shots_against_season",
    "away_shots_against_season",

    "home_shooting_pct_season",
    "away_shooting_pct_season",
    "home_save_pct_season",
    "away_save_pct_season",

    "home_power_play_pct_season",
    "away_power_play_pct_season",

    "home_days_rest",
    "away_days_rest",
    "home_back_to_back",
    "away_back_to_back",

    "rest_diff"
]


def main():
    data = pd.read_csv(
        INPUT_FILE
    )

    train = data[
        data["season"].isin(
            TRAIN_SEASONS
        )
    ].copy()

    X_train = train[
        FEATURES
    ]

    y_train = train[
        "total_goals"
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
            RandomForestRegressor(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=8,
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
            "features": FEATURES,
            "train_seasons": TRAIN_SEASONS
        },
        MODEL_FILE
    )

    print(
        "NHL Total Production Model"
    )

    print(
        "=========================="
    )

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
        f"{len(FEATURES)}"
    )

    print()

    print(
        f"Saved: "
        f"{MODEL_FILE}"
    )


if __name__ == "__main__":
    main()