from itertools import product
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

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

N_ESTIMATORS = [100, 200, 300]
LEARNING_RATES = [0.01, 0.03, 0.05]
MAX_DEPTHS = [1, 2, 3]
MIN_SAMPLES_LEAF = [5, 10, 20]

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

    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train)
    X_validation = imputer.transform(X_validation)

    combinations = list(
        product(
            N_ESTIMATORS,
            LEARNING_RATES,
            MAX_DEPTHS,
            MIN_SAMPLES_LEAF
        )
    )

    print("NHL Puck Line Gradient Boosting Tuning")
    print("======================================")
    print(f"Train seasons: {TRAIN_SEASONS}")
    print(f"Validation season: {VALIDATION_SEASON}")
    print(f"Training games: {len(train)}")
    print(f"Validation games: {len(validation)}")
    print(f"Features: {len(FEATURES)}")
    print(f"Configurations: {len(combinations)}")
    print()

    results = []

    for index, (
        n_estimators,
        learning_rate,
        max_depth,
        min_samples_leaf
    ) in enumerate(combinations, start=1):
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            loss="absolute_error",
            random_state=42
        )

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
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "mae": mae,
            "winner_accuracy": winner_accuracy,
            "prediction_mean": predictions.mean(),
            "prediction_std": predictions.std()
        })

        print(
            f"{index:>2}/{len(combinations)} "
            f"trees={n_estimators} "
            f"rate={learning_rate} "
            f"depth={max_depth} "
            f"leaf={min_samples_leaf} "
            f"mae={mae:.3f} "
            f"winner={winner_accuracy:.3f}"
        )

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
    print("TOP 15 CONFIGURATIONS")
    print("=====================")

    print(
        results.head(15).to_string(
            index=False,
            formatters={
                "learning_rate": "{:.3f}".format,
                "mae": "{:.3f}".format,
                "winner_accuracy": "{:.3f}".format,
                "prediction_mean": "{:.3f}".format,
                "prediction_std": "{:.3f}".format
            }
        )
    )

    best = results.iloc[0]

    print()
    print("BEST CONFIGURATION")
    print("==================")
    print(
        f"n_estimators: "
        f"{int(best['n_estimators'])}"
    )
    print(
        f"learning_rate: "
        f"{best['learning_rate']:.3f}"
    )
    print(
        f"max_depth: "
        f"{int(best['max_depth'])}"
    )
    print(
        f"min_samples_leaf: "
        f"{int(best['min_samples_leaf'])}"
    )
    print(
        f"MAE: "
        f"{best['mae']:.3f}"
    )
    print(
        f"Winner accuracy: "
        f"{best['winner_accuracy']:.3f}"
    )
    print(
        f"Improvement over baseline MAE: "
        f"{baseline_mae - best['mae']:+.3f}"
    )

if __name__ == "__main__":
    main()