from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, log_loss

INPUT_FILE = Path("data/processed/nhl_features.csv")

TRAIN_SEASONS = [2022, 2023, 2024]
VALIDATION_SEASON = 2025

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

def main():
    data = pd.read_csv(INPUT_FILE)

    train = data[
        data["season"].isin(TRAIN_SEASONS)
    ].copy()

    validation = data[
        data["season"] == VALIDATION_SEASON
    ].copy()

    X_train = train[FEATURES]
    y_train = train["home_win"]

    X_validation = validation[FEATURES]
    y_validation = validation["home_win"]

    imputer = SimpleImputer(strategy="median")

    X_train_imputed = imputer.fit_transform(X_train)
    X_validation_imputed = imputer.transform(X_validation)

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_imputed,
        y_train
    )

    predictions = model.predict(
        X_validation_imputed
    )

    probabilities = model.predict_proba(
        X_validation_imputed
    )[:, 1]

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    loss = log_loss(
        y_validation,
        probabilities
    )

    correct = int(
        (
            predictions
            == y_validation
        ).sum()
    )

    permutation = permutation_importance(
        model,
        X_validation_imputed,
        y_validation,
        scoring="neg_log_loss",
        n_repeats=20,
        random_state=42,
        n_jobs=-1
    )

    results = pd.DataFrame({
        "feature": FEATURES,
        "rf_importance": model.feature_importances_,
        "permutation_importance": permutation.importances_mean,
        "permutation_std": permutation.importances_std
    })

    results = results.sort_values(
        "permutation_importance",
        ascending=False
    ).reset_index(drop=True)

    print("NHL Winner Feature Analysis")
    print("===========================")
    print(f"Train seasons: {TRAIN_SEASONS}")
    print(f"Validation season: {VALIDATION_SEASON}")
    print(f"Training games: {len(train)}")
    print(f"Validation games: {len(validation)}")
    print(f"Features: {len(FEATURES)}")

    print()
    print("VALIDATION PERFORMANCE")
    print("======================")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Correct: {correct}/{len(validation)}")
    print(f"Log loss: {loss:.3f}")

    print()
    print("FEATURE IMPORTANCE")
    print("==================")
    print(
        results.to_string(
            index=False,
            formatters={
                "rf_importance": "{:.5f}".format,
                "permutation_importance": "{:+.5f}".format,
                "permutation_std": "{:.5f}".format
            }
        )
    )

    positive = results[
        results["permutation_importance"] > 0
    ]

    neutral = results[
        results["permutation_importance"] == 0
    ]

    negative = results[
        results["permutation_importance"] < 0
    ]

    print()
    print("SUMMARY")
    print("=======")
    print(
        f"Positive permutation importance: "
        f"{len(positive)}"
    )
    print(
        f"Zero permutation importance: "
        f"{len(neutral)}"
    )
    print(
        f"Negative permutation importance: "
        f"{len(negative)}"
    )

    print()
    print("TOP 15")
    print("======")
    print(
        results.head(15)[
            [
                "feature",
                "permutation_importance",
                "rf_importance"
            ]
        ].to_string(
            index=False,
            formatters={
                "permutation_importance": "{:+.5f}".format,
                "rf_importance": "{:.5f}".format
            }
        )
    )

    print()
    print("BOTTOM 15")
    print("=========")
    print(
        results.tail(15)[
            [
                "feature",
                "permutation_importance",
                "rf_importance"
            ]
        ].to_string(
            index=False,
            formatters={
                "permutation_importance": "{:+.5f}".format,
                "rf_importance": "{:.5f}".format
            }
        )
    )

if __name__ == "__main__":
    main()