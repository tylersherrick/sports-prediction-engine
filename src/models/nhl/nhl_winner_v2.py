from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline

INPUT_FILE = Path("data/processed/nhl_features.csv")

TRAIN_SEASONS = [2022, 2023, 2024, 2025]
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

def main():
    data = pd.read_csv(INPUT_FILE)

    train = data[
        data["season"].isin(TRAIN_SEASONS)
    ].copy()

    test = data[
        data["season"] == TEST_SEASON
    ].copy()

    X_train = train[FEATURES]
    y_train = train["home_win"]

    X_test = test[FEATURES]
    y_test = test["home_win"]

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
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

    correct = int(
        (
            predictions
            == y_test
        ).sum()
    )

    baseline_accuracy = y_test.mean()

    print("NHL Winner Model V2")
    print("===================")
    print(f"Train seasons: {TRAIN_SEASONS}")
    print(f"Test season: {TEST_SEASON}")
    print(f"Training games: {len(train)}")
    print(f"Test games: {len(test)}")
    print(f"Features: {len(FEATURES)}")

    print()
    print("Locked configuration:")
    print("n_estimators: 500")
    print("max_depth: 4")
    print("min_samples_leaf: 5")
    print("max_features: sqrt")

    print()
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Correct: {correct}/{len(test)}")
    print(f"Log loss: {loss:.3f}")
    print(
        f"Home-team baseline: "
        f"{baseline_accuracy:.3f}"
    )
    print(
        f"Vs baseline: "
        f"{accuracy - baseline_accuracy:+.3f}"
    )

    print()
    print("V1 comparison:")
    print("V1 accuracy: 0.540")
    print("V1 correct: 709/1312")
    print(
        f"V2 accuracy: {accuracy:.3f}"
    )
    print(
        f"V2 correct: {correct}/{len(test)}"
    )
    print(
        f"Accuracy change: "
        f"{accuracy - 0.540:+.3f}"
    )
    print(
        f"Additional correct games: "
        f"{correct - 709:+d}"
    )

if __name__ == "__main__":
    main()