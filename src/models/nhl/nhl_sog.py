from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

INPUT_FILE = Path(
    "data/processed/nhl_skater_features.csv"
)

TRAIN_SEASONS = [
    2022,
    2023,
    2024,
    2025
]

TEST_SEASON = 2026

MIN_PRIOR_GAMES = 5

FEATURES = [
    "home",
    "player_games_played",

    "shots_on_goal_last_5",
    "shots_on_goal_last_10",
    "shots_on_goal_season",

    "shots_missed_last_5",
    "shots_missed_last_10",
    "shots_missed_season",

    "goals_last_5",
    "goals_last_10",
    "goals_season",

    "assists_last_5",
    "assists_last_10",
    "assists_season",

    "points_last_5",
    "points_last_10",
    "points_season",

    "time_on_ice_last_5",
    "time_on_ice_last_10",
    "time_on_ice_season",

    "power_play_time_on_ice_last_5",
    "power_play_time_on_ice_last_10",
    "power_play_time_on_ice_season",

    "even_strength_time_on_ice_last_5",
    "even_strength_time_on_ice_last_10",
    "even_strength_time_on_ice_season",

    "shifts_last_5",
    "shifts_last_10",
    "shifts_season",

    "shots_per_60_last_5",
    "pp_usage_share_last_5",

    "days_rest",
    "back_to_back",

    "opponent_shots_against_last_5",
    "opponent_shots_against_season",
    "opponent_goals_against_last_5",
    "opponent_goals_against_season",
    "opponent_save_pct_last_5",
    "opponent_save_pct_season",
    "opponent_penalty_minutes_last_5",
    "opponent_penalty_minutes_season",
    "opponent_days_rest",
    "opponent_back_to_back"
]

def main():
    data = pd.read_csv(
        INPUT_FILE
    )

    data = data[
        data[
            "player_games_played"
        ] >= MIN_PRIOR_GAMES
    ].copy()

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
        "shots_on_goal"
    ]

    X_test = test[
        FEATURES
    ]

    y_test = test[
        "shots_on_goal"
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
                max_depth=8,
                min_samples_leaf=10,
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
            "shots_on_goal_season"
        ]
        .fillna(
            train[
                "shots_on_goal"
            ].mean()
        )
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )

    print(
        "NHL Shots On Goal Model"
    )

    print(
        "======================="
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
        f"\nSOG MAE: "
        f"{mae:.3f}"
    )

    print(
        f"Player-average baseline MAE: "
        f"{baseline_mae:.3f}"
    )

    results = test[
        [
            "date",
            "player",
            "team",
            "opponent",
            "position",
            "shots_on_goal",
            "shots_on_goal_last_5",
            "shots_on_goal_last_10",
            "shots_on_goal_season"
        ]
    ].copy()

    results[
        "predicted_sog"
    ] = predictions

    results[
        "error"
    ] = (
        results[
            "predicted_sog"
        ]
        - results[
            "shots_on_goal"
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