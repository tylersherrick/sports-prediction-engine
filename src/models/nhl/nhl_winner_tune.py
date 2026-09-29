from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss

INPUT_FILE = Path(
    "data/processed/nhl_features.csv"
)

ROLLING_WINDOWS = [
    5,
    10,
    15
]

ROLLING_STATS = [
    "goals_for",
    "goals_against",
    "goal_diff",
    "shots_for",
    "shots_against",
    "shot_diff",
    "shooting_pct",
    "save_pct",
    "power_play_pct",
    "penalty_minutes",
    "win"
]

SEASON_STATS = [
    "goals_for",
    "goals_against",
    "goal_diff",
    "shots_for",
    "shots_against",
    "shot_diff",
    "shooting_pct",
    "save_pct",
    "power_play_pct",
    "win"
]

VENUE_STATS = [
    "goals_for",
    "goals_against",
    "goal_diff",
    "shots_for",
    "shots_against",
    "shot_diff",
    "shooting_pct",
    "save_pct",
    "power_play_pct",
    "win"
]

CONTEXT_FEATURES = [
    "home_days_rest",
    "away_days_rest",
    "home_back_to_back",
    "away_back_to_back",
    "rest_diff",
    "games_played_diff",
    "home_games_played"
]

ROAD_FEATURES = [
    "home_road_streak",
    "away_road_streak",
    "road_streak_diff",
    "home_home_after_road_trip",
    "away_home_after_road_trip",
    "home_after_road_trip_diff"
]

SPLITS = [
    (
        [2016, 2017, 2018, 2019, 2020],
        2021
    ),
    (
        [2016, 2017, 2018, 2019, 2020, 2021],
        2022
    ),
    (
        [2016, 2017, 2018, 2019, 2020, 2021, 2022],
        2023
    ),
    (
        [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
        2024
    ),
    (
        [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        2025
    ),
    (
        [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        2026
    )
]

CONFIG = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_leaf": 5,
    "max_features": "sqrt"
}


def build_features():
    features = []

    for stat in SEASON_STATS:
        features.extend([
            f"home_{stat}_season",
            f"away_{stat}_season",
            f"{stat}_season_diff"
        ])

    for stat in VENUE_STATS:
        features.extend([
            f"home_{stat}_venue_season",
            f"away_{stat}_venue_season",
            f"{stat}_venue_season_diff"
        ])

    for window in ROLLING_WINDOWS:
        for stat in ROLLING_STATS:
            features.extend([
                f"home_{stat}_last_{window}",
                f"away_{stat}_last_{window}",
                f"{stat}_last_{window}_diff"
            ])

    features.extend(
        CONTEXT_FEATURES
    )

    features.extend(
        ROAD_FEATURES
    )

    return features


def evaluate(
    data,
    train_seasons,
    validation_season,
    features
):
    train = data[
        data["season"].isin(
            train_seasons
        )
    ].copy()

    validation = data[
        data["season"]
        == validation_season
    ].copy()

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(
        train[features]
    )

    X_validation = imputer.transform(
        validation[features]
    )

    y_train = train[
        "home_win"
    ]

    y_validation = validation[
        "home_win"
    ]

    model = RandomForestClassifier(
        n_estimators=CONFIG[
            "n_estimators"
        ],
        max_depth=CONFIG[
            "max_depth"
        ],
        min_samples_leaf=CONFIG[
            "min_samples_leaf"
        ],
        max_features=CONFIG[
            "max_features"
        ],
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_validation
    )

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    home_baseline_predictions = (
        pd.Series(
            1,
            index=y_validation.index
        )
    )

    return {
        "train_games": len(
            train
        ),
        "accuracy": accuracy_score(
            y_validation,
            predictions
        ),
        "correct": int(
            (
                predictions
                == y_validation
            ).sum()
        ),
        "games": len(
            validation
        ),
        "log_loss": log_loss(
            y_validation,
            probabilities
        ),
        "home_baseline": accuracy_score(
            y_validation,
            home_baseline_predictions
        )
    }


def main():
    data = pd.read_csv(
        INPUT_FILE
    )

    features = build_features()

    print(
        "NHL Winner Final Expanded "
        "Walk-Forward Test"
    )

    print(
        "======================================"
    )

    print(
        f"Dataset games: {len(data)}"
    )

    print(
        f"Dataset seasons: "
        f"{sorted(data['season'].unique())}"
    )

    print(
        f"Features: {len(features)}"
    )

    print(
        "Model: "
        "RandomForestClassifier("
        "trees=500, depth=4, "
        "leaf=5, max_features=sqrt)"
    )

    print()

    total_correct = 0
    total_games = 0
    weighted_log_loss = 0.0
    baseline_correct = 0
    season_accuracies = []

    results = []

    for (
        train_seasons,
        validation_season
    ) in SPLITS:
        result = evaluate(
            data,
            train_seasons,
            validation_season,
            features
        )

        total_correct += result[
            "correct"
        ]

        total_games += result[
            "games"
        ]

        weighted_log_loss += (
            result["log_loss"]
            * result["games"]
        )

        baseline_correct += (
            result["home_baseline"]
            * result["games"]
        )

        season_accuracies.append(
            result["accuracy"]
        )

        results.append({
            "season": validation_season,
            **result
        })

        print(
            f"{validation_season} | "
            f"train_games="
            f"{result['train_games']} | "
            f"accuracy="
            f"{result['accuracy']:.3f} | "
            f"correct="
            f"{result['correct']}/"
            f"{result['games']} | "
            f"log_loss="
            f"{result['log_loss']:.3f} | "
            f"home_baseline="
            f"{result['home_baseline']:.3f}"
        )

    overall_accuracy = (
        total_correct
        / total_games
    )

    overall_log_loss = (
        weighted_log_loss
        / total_games
    )

    overall_home_baseline = (
        baseline_correct
        / total_games
    )

    print()

    print(
        "FINAL EXPANDED RESULT"
    )

    print(
        "====================="
    )

    print(
        f"Accuracy: "
        f"{overall_accuracy:.3f}"
    )

    print(
        f"Correct: "
        f"{total_correct}/"
        f"{total_games}"
    )

    print(
        f"Log loss: "
        f"{overall_log_loss:.3f}"
    )

    print(
        f"Home-team baseline: "
        f"{overall_home_baseline:.3f}"
    )

    print(
        f"Vs baseline: "
        f"{overall_accuracy - overall_home_baseline:+.3f}"
    )

    print(
        f"Best season: "
        f"{max(season_accuracies):.3f}"
    )

    print(
        f"Worst season: "
        f"{min(season_accuracies):.3f}"
    )

    print()

    print(
        "SEASON RESULTS"
    )

    print(
        "=============="
    )

    results_df = pd.DataFrame(
        results
    )

    print(
        results_df[
            [
                "season",
                "train_games",
                "accuracy",
                "correct",
                "games",
                "log_loss",
                "home_baseline"
            ]
        ].to_string(
            index=False,
            formatters={
                "accuracy":
                    "{:.3f}".format,
                "log_loss":
                    "{:.3f}".format,
                "home_baseline":
                    "{:.3f}".format
            }
        )
    )


if __name__ == "__main__":
    main()