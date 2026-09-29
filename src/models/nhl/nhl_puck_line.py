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
    "home_goal_diff_last_5",
    "away_goal_diff_last_5",

    "home_shots_for_last_5",
    "away_shots_for_last_5",
    "home_shots_against_last_5",
    "away_shots_against_last_5",
    "home_shot_diff_last_5",
    "away_shot_diff_last_5",

    "home_shooting_pct_last_5",
    "away_shooting_pct_last_5",
    "home_save_pct_last_5",
    "away_save_pct_last_5",
    "home_power_play_pct_last_5",
    "away_power_play_pct_last_5",

    "home_win_last_5",
    "away_win_last_5",

    "home_goals_for_season",
    "away_goals_for_season",
    "home_goals_against_season",
    "away_goals_against_season",
    "home_goal_diff_season",
    "away_goal_diff_season",

    "home_shots_for_season",
    "away_shots_for_season",
    "home_shots_against_season",
    "away_shots_against_season",
    "home_shot_diff_season",
    "away_shot_diff_season",

    "home_shooting_pct_season",
    "away_shooting_pct_season",
    "home_save_pct_season",
    "away_save_pct_season",
    "home_power_play_pct_season",
    "away_power_play_pct_season",

    "home_win_season",
    "away_win_season",

    "home_days_rest",
    "away_days_rest",
    "home_back_to_back",
    "away_back_to_back",

    "goal_diff_last_5_diff",
    "shot_diff_last_5_diff",
    "shooting_pct_last_5_diff",
    "save_pct_last_5_diff",
    "power_play_pct_last_5_diff",
    "win_last_5_diff",

    "goal_diff_season_diff",
    "shot_diff_season_diff",
    "shooting_pct_season_diff",
    "save_pct_season_diff",
    "power_play_pct_season_diff",
    "win_season_diff",

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
        "home_goal_diff"
    ]

    X_test = test[
        FEATURES
    ]

    y_test = test[
        "home_goal_diff"
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

    winner_predictions = (
        predictions > 0
    ).astype(int)

    actual_winners = (
        y_test > 0
    ).astype(int)

    winner_correct = (
        winner_predictions
        == actual_winners
    ).sum()

    winner_accuracy = (
        winner_correct
        / len(test)
    )

    print(
        "NHL Puck Line Model"
    )

    print(
        "==================="
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
        f"\nGoal-margin MAE: "
        f"{mae:.3f}"
    )

    print(
        f"Winner from margin: "
        f"{winner_accuracy:.3f}"
    )

    print(
        f"Winner correct: "
        f"{winner_correct}/"
        f"{len(test)}"
    )

    results = test[
        [
            "date",
            "away_team",
            "home_team",
            "away_score",
            "home_score",
            "home_goal_diff"
        ]
    ].copy()

    results[
        "predicted_home_margin"
    ] = predictions

    results[
        "margin_error"
    ] = (
        results[
            "predicted_home_margin"
        ]
        - results[
            "home_goal_diff"
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