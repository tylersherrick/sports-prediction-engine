from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

INPUT_FILE = Path(
    "data/processed/nhl_features.csv"
)

TRAIN_SEASONS = [
    2022,
    2023,
    2024,
    2025
]

TEST_SEASON = 2026

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

    test = data[
        data["season"]
        == TEST_SEASON
    ].copy()

    X_train = train[
        FEATURES
    ]

    y_train = train[
        "total_goals"
    ]

    X_test = test[
        FEATURES
    ]

    y_test = test[
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

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    baseline_prediction = (
        y_train.mean()
    )

    baseline_predictions = [
        baseline_prediction
    ] * len(y_test)

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )

    print(
        "NHL Total Model"
    )

    print(
        "==============="
    )

    print(
        f"Train seasons: "
        f"{TRAIN_SEASONS}"
    )

    print(
        f"Test season: "
        f"{TEST_SEASON}"
    )

    print(
        f"Training games: "
        f"{len(train)}"
    )

    print(
        f"Test games: "
        f"{len(test)}"
    )

    print(
        f"Features: "
        f"{len(FEATURES)}"
    )

    print(
        f"\nTotal-goals MAE: "
        f"{mae:.3f}"
    )

    print(
        f"Baseline MAE: "
        f"{baseline_mae:.3f}"
    )

    print(
        f"Training average total: "
        f"{baseline_prediction:.3f}"
    )

    results = test[
        [
            "date",
            "away_team",
            "home_team",
            "away_score",
            "home_score",
            "total_goals"
        ]
    ].copy()

    results[
        "predicted_total"
    ] = predictions

    results[
        "total_error"
    ] = (
        results[
            "predicted_total"
        ]
        - results[
            "total_goals"
        ]
    ).abs()

    print(
        "\nLast 20 predictions:"
    )

    print(
        results
        .tail(20)
        .to_string(
            index=False
        )
    )

    importances = (
        model
        .named_steps["model"]
        .feature_importances_
    )

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": importances
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print(
        "\nTop 20 features:"
    )

    print(
        importance
        .head(20)
        .to_string(
            index=False
        )
    )

if __name__ == "__main__":
    main()