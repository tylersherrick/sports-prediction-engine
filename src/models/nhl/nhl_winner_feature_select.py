from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, log_loss

INPUT_FILE = Path("data/processed/nhl_features.csv")

RANK_TRAIN_SEASONS = [2022, 2023]
RANK_VALIDATION_SEASON = 2024

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

FEATURE_COUNTS = [10, 15, 20, 25, 29, 35, 45, 57]

def build_model():
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

def main():
    data = pd.read_csv(INPUT_FILE)

    rank_train = data[
        data["season"].isin(RANK_TRAIN_SEASONS)
    ].copy()

    rank_validation = data[
        data["season"] == RANK_VALIDATION_SEASON
    ].copy()

    rank_imputer = SimpleImputer(strategy="median")

    X_rank_train = rank_imputer.fit_transform(
        rank_train[FEATURES]
    )

    X_rank_validation = rank_imputer.transform(
        rank_validation[FEATURES]
    )

    rank_model = build_model()

    rank_model.fit(
        X_rank_train,
        rank_train["home_win"]
    )

    permutation = permutation_importance(
        rank_model,
        X_rank_validation,
        rank_validation["home_win"],
        scoring="neg_log_loss",
        n_repeats=20,
        random_state=42,
        n_jobs=-1
    )

    ranking = pd.DataFrame({
        "feature": FEATURES,
        "importance": permutation.importances_mean
    }).sort_values(
        "importance",
        ascending=False
    ).reset_index(drop=True)

    train = data[
        data["season"].isin(TRAIN_SEASONS)
    ].copy()

    validation = data[
        data["season"] == VALIDATION_SEASON
    ].copy()

    results = []

    print("NHL Winner Feature Selection")
    print("============================")
    print(
        f"Feature ranking: "
        f"{RANK_TRAIN_SEASONS} -> {RANK_VALIDATION_SEASON}"
    )
    print(
        f"Model validation: "
        f"{TRAIN_SEASONS} -> {VALIDATION_SEASON}"
    )
    print()

    print("FEATURE RANKING")
    print("===============")

    print(
        ranking.to_string(
            index=True,
            formatters={
                "importance": "{:+.5f}".format
            }
        )
    )

    print()
    print("FEATURE COUNT TESTS")
    print("===================")

    for count in FEATURE_COUNTS:
        selected_features = ranking.head(count)[
            "feature"
        ].tolist()

        imputer = SimpleImputer(
            strategy="median"
        )

        X_train = imputer.fit_transform(
            train[selected_features]
        )

        X_validation = imputer.transform(
            validation[selected_features]
        )

        model = build_model()

        model.fit(
            X_train,
            train["home_win"]
        )

        predictions = model.predict(
            X_validation
        )

        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        accuracy = accuracy_score(
            validation["home_win"],
            predictions
        )

        loss = log_loss(
            validation["home_win"],
            probabilities
        )

        correct = int(
            (
                predictions
                == validation["home_win"]
            ).sum()
        )

        results.append({
            "features": count,
            "accuracy": accuracy,
            "correct": correct,
            "log_loss": loss
        })

        print(
            f"{count:>2} features | "
            f"accuracy={accuracy:.3f} | "
            f"correct={correct}/{len(validation)} | "
            f"log_loss={loss:.3f}"
        )

    results = pd.DataFrame(results)

    best_accuracy = results.sort_values(
        ["accuracy", "log_loss"],
        ascending=[False, True]
    ).iloc[0]

    best_loss = results.sort_values(
        ["log_loss", "accuracy"],
        ascending=[True, False]
    ).iloc[0]

    print()
    print("BEST ACCURACY")
    print("=============")
    print(
        f"Features: "
        f"{int(best_accuracy['features'])}"
    )
    print(
        f"Accuracy: "
        f"{best_accuracy['accuracy']:.3f}"
    )
    print(
        f"Correct: "
        f"{int(best_accuracy['correct'])}/{len(validation)}"
    )
    print(
        f"Log loss: "
        f"{best_accuracy['log_loss']:.3f}"
    )

    print()
    print("BEST LOG LOSS")
    print("=============")
    print(
        f"Features: "
        f"{int(best_loss['features'])}"
    )
    print(
        f"Accuracy: "
        f"{best_loss['accuracy']:.3f}"
    )
    print(
        f"Correct: "
        f"{int(best_loss['correct'])}/{len(validation)}"
    )
    print(
        f"Log loss: "
        f"{best_loss['log_loss']:.3f}"
    )

if __name__ == "__main__":
    main()