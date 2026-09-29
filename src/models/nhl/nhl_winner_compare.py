from pathlib import Path

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

    "home_win_last_5",
    "away_win_last_5",
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

    "goal_diff_season_diff",
    "shot_diff_season_diff",
    "shooting_pct_season_diff",
    "save_pct_season_diff",
    "power_play_pct_season_diff",

    "rest_diff",
    "games_played_diff",
    "home_games_played"
]

def build_models():
    return {
        "Logistic Regression": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            )
        ]),

        "Random Forest": Pipeline([
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
                    max_depth=6,
                    min_samples_leaf=8,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]),

        "Gradient Boosting": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.03,
                    max_depth=2,
                    min_samples_leaf=10,
                    random_state=42
                )
            )
        ]),

        "HistGradientBoosting": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=200,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=42
                )
            )
        ])
    }

def main():
    data = pd.read_csv(
        INPUT_FILE
    )

    train = data[
        data[
            "season"
        ].isin(
            TRAIN_SEASONS
        )
    ].copy()

    test = data[
        data[
            "season"
        ] == TEST_SEASON
    ].copy()

    X_train = train[
        FEATURES
    ]

    y_train = train[
        "home_win"
    ]

    X_test = test[
        FEATURES
    ]

    y_test = test[
        "home_win"
    ]

    print(
        "NHL Winner Model Comparison"
    )

    print(
        "==========================="
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

    baseline_predictions = pd.Series(
        1,
        index=test.index
    )

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions
    )

    print(
        f"\nHome-team baseline accuracy: "
        f"{baseline_accuracy:.3f}"
    )

    models = build_models()

    results = []

    for name, model in models.items():
        print(
            f"\nTraining {name}..."
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        loss = log_loss(
            y_test,
            probabilities
        )

        results.append({
            "model": name,
            "accuracy": accuracy,
            "correct": int(
                (
                    predictions
                    == y_test
                ).sum()
            ),
            "games": len(y_test),
            "log_loss": loss,
            "vs_baseline": (
                accuracy
                - baseline_accuracy
            )
        })

    results = pd.DataFrame(
        results
    )

    results = results.sort_values(
        [
            "accuracy",
            "log_loss"
        ],
        ascending=[
            False,
            True
        ]
    )

    print()
    print(
        "RESULTS"
    )

    print(
        "======="
    )

    print(
        results.to_string(
            index=False,
            formatters={
                "accuracy": "{:.3f}".format,
                "log_loss": "{:.3f}".format,
                "vs_baseline": "{:+.3f}".format
            }
        )
    )

if __name__ == "__main__":
    main()