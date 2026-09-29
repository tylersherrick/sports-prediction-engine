from pathlib import Path

import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

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

def build_models():
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=10,
            loss="absolute_error",
            random_state=42
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            loss="absolute_error",
            random_state=42
        )
    }

def main():
    data = pd.read_csv(INPUT_FILE)

    train = data[
        data["season"].isin(TRAIN_SEASONS)
    ].copy()

    validation = data[
        data["season"] == VALIDATION_SEASON
    ].copy()

    X_train = train[FEATURES]
    y_train = train["home_goal_diff"]

    X_validation = validation[FEATURES]
    y_validation = validation["home_goal_diff"]

    print("NHL Puck Line Model Comparison")
    print("==============================")
    print(f"Train seasons: {TRAIN_SEASONS}")
    print(f"Validation season: {VALIDATION_SEASON}")
    print(f"Training games: {len(train)}")
    print(f"Validation games: {len(validation)}")
    print(f"Features: {len(FEATURES)}")

    models = build_models()

    results = []

    for name, estimator in models.items():
        print()
        print(f"Training {name}...")

        model = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                estimator
            )
        ])

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_validation
        )

        mae = mean_absolute_error(
            y_validation,
            predictions
        )

        winner_predictions = (
            predictions > 0
        ).astype(int)

        actual_winners = (
            y_validation > 0
        ).astype(int)

        winner_accuracy = (
            winner_predictions
            == actual_winners
        ).mean()

        results.append({
            "model": name,
            "mae": mae,
            "winner_accuracy": winner_accuracy,
            "prediction_mean": predictions.mean(),
            "prediction_std": predictions.std()
        })

    results = pd.DataFrame(results)

    results = results.sort_values(
        ["mae", "winner_accuracy"],
        ascending=[True, False]
    )

    baseline_prediction = y_train.median()

    baseline_predictions = pd.Series(
        baseline_prediction,
        index=y_validation.index
    )

    baseline_mae = mean_absolute_error(
        y_validation,
        baseline_predictions
    )

    print()
    print("BASELINE")
    print("========")
    print(
        f"Training median goal margin: "
        f"{baseline_prediction:.3f}"
    )
    print(
        f"Validation baseline MAE: "
        f"{baseline_mae:.3f}"
    )

    print()
    print("RESULTS")
    print("=======")

    print(
        results.to_string(
            index=False,
            formatters={
                "mae": "{:.3f}".format,
                "winner_accuracy": "{:.3f}".format,
                "prediction_mean": "{:.3f}".format,
                "prediction_std": "{:.3f}".format
            }
        )
    )

    best = results.iloc[0]

    print()
    print("BEST MODEL")
    print("==========")
    print(f"Model: {best['model']}")
    print(f"MAE: {best['mae']:.3f}")
    print(
        f"Winner accuracy from margin: "
        f"{best['winner_accuracy']:.3f}"
    )
    print(
        f"Improvement over baseline MAE: "
        f"{baseline_mae - best['mae']:+.3f}"
    )

if __name__ == "__main__":
    main()