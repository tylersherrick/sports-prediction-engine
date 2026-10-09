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
    2016, 2017, 2018, 2019, 2020,
    2021, 2022, 2023, 2024, 2025
]

TEST_SEASON = 2026

COMMON_FEATURES = [
    "home",
    "player_games_played",
    "goals_last_5",
    "goals_last_10",
    "goals_season",
    "shots_on_goal_last_5",
    "shots_on_goal_last_10",
    "shots_on_goal_season",
    "shots_missed_last_5",
    "shots_missed_last_10",
    "shots_missed_season",
    "points_last_5",
    "points_last_10",
    "points_season",
    "assists_last_5",
    "assists_last_10",
    "assists_season",
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
    "opponent_goals_against_last_5",
    "opponent_goals_against_season",
    "opponent_shots_against_last_5",
    "opponent_shots_against_season",
    "opponent_save_pct_last_5",
    "opponent_save_pct_season",
    "opponent_penalty_minutes_last_5",
    "opponent_penalty_minutes_season",
    "opponent_days_rest",
    "opponent_back_to_back"
]

PROPS = {
    "Points": "points",
    "Goals": "goals"
}


def build_model():
    return Pipeline([
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


def run_test(
    data,
    target,
    min_prior_games
):
    train = data[
        data["season"].isin(
            TRAIN_SEASONS
        )
        & (
            data["player_games_played"]
            >= min_prior_games
        )
    ].copy()

    test = data[
        (data["season"] == TEST_SEASON)
        & (
            data["player_games_played"]
            >= min_prior_games
        )
    ].copy()

    model = build_model()

    model.fit(
        train[COMMON_FEATURES],
        train[target]
    )

    predictions = model.predict(
        test[COMMON_FEATURES]
    )

    mae = mean_absolute_error(
        test[target],
        predictions
    )

    return (
        len(train),
        len(test),
        mae
    )


def main():
    data = pd.read_csv(
        INPUT_FILE
    )

    print()
    print(
        "NHL Skater Early-Season Comparison"
    )

    print(
        "=================================="
    )

    for name, target in PROPS.items():
        print()
        print(name)
        print("-" * len(name))

        for minimum in [
            5,
            0
        ]:
            (
                train_rows,
                test_rows,
                mae
            ) = run_test(
                data,
                target,
                minimum
            )

            print(
                f"Min games {minimum}: "
                f"MAE {mae:.3f} | "
                f"Train {train_rows} | "
                f"Test {test_rows}"
            )


if __name__ == "__main__":
    main()