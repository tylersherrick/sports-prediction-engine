from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

INPUT_FILE = Path(
    "data/processed/nhl_goalie_features.csv"
)

TRAIN_SEASONS = [
    2022,
    2023,
    2024,
    2025
]

TEST_SEASON = 2026

MIN_PRIOR_GAMES = 5
MIN_TARGET_TOI = 2700

FEATURES = [
    "home",
    "goalie_games_played",

    "saves_last_5",
    "saves_last_10",
    "saves_season",

    "shots_against_last_5",
    "shots_against_last_10",
    "shots_against_season",

    "goals_against_last_5",
    "goals_against_last_10",
    "goals_against_season",

    "save_pct_last_5",
    "save_pct_last_10",
    "save_pct_season",

    "even_strength_saves_last_5",
    "even_strength_saves_last_10",
    "even_strength_saves_season",

    "power_play_saves_last_5",
    "power_play_saves_last_10",
    "power_play_saves_season",

    "shorthanded_saves_last_5",
    "shorthanded_saves_last_10",
    "shorthanded_saves_season",

    "time_on_ice_last_5",
    "time_on_ice_last_10",
    "time_on_ice_season",

    "saves_per_60_last_5",
    "shots_against_per_60_last_5",

    "days_rest",
    "back_to_back",

    "opponent_shots_for_last_5",
    "opponent_shots_for_season",

    "opponent_goals_for_last_5",
    "opponent_goals_for_season",

    "opponent_shooting_pct_last_5",
    "opponent_shooting_pct_season",

    "opponent_power_play_pct_last_5",
    "opponent_power_play_pct_season",

    "opponent_days_rest",
    "opponent_back_to_back"
]

def main():
    data = pd.read_csv(
        INPUT_FILE
    )

    eligible = data[
        (
            data[
                "goalie_games_played"
            ] >= MIN_PRIOR_GAMES
        )
        &
        (
            data[
                "time_on_ice"
            ] >= MIN_TARGET_TOI
        )
    ].copy()

    train = eligible[
        eligible[
            "season"
        ].isin(
            TRAIN_SEASONS
        )
    ].copy()

    test = eligible[
        eligible[
            "season"
        ] == TEST_SEASON
    ].copy()

    X_train = train[
        FEATURES
    ]

    y_train = train[
        "saves"
    ]

    X_test = test[
        FEATURES
    ]

    y_test = test[
        "saves"
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
                max_depth=7,
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

    baseline_predictions = (
        test[
            "saves_season"
        ]
        .fillna(
            train[
                "saves"
            ].mean()
        )
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )

    print(
        "NHL Goalie Saves Model"
    )

    print(
        "======================"
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
        f"Minimum prior games: "
        f"{MIN_PRIOR_GAMES}"
    )

    print(
        f"Minimum target TOI: "
        f"{MIN_TARGET_TOI} seconds"
    )

    print(
        f"Training rows: "
        f"{len(train)}"
    )

    print(
        f"Test rows: "
        f"{len(test)}"
    )

    print(
        f"Features: "
        f"{len(FEATURES)}"
    )

    print(
        f"\nSaves MAE: "
        f"{mae:.3f}"
    )

    print(
        f"Goalie-average baseline MAE: "
        f"{baseline_mae:.3f}"
    )

    results = test[
        [
            "date",
            "player",
            "team",
            "opponent",
            "saves",
            "saves_last_5",
            "saves_last_10",
            "saves_season",
            "opponent_shots_for_last_5"
        ]
    ].copy()

    results[
        "predicted_saves"
    ] = predictions

    results[
        "error"
    ] = (
        results[
            "predicted_saves"
        ]
        - results[
            "saves"
        ]
    ).abs()

    print(
        "\nLast 30 predictions:"
    )

    print(
        results
        .tail(30)
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

    print(
        "\nPrediction distribution:"
    )

    print(
        pd.Series(
            predictions
        )
        .describe()
    )

if __name__ == "__main__":
    main()