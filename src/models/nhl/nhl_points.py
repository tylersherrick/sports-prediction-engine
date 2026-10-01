from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


INPUT_FILE = Path("data/processed/nhl_skater_features.csv")
MODEL_FILE = Path("models/nhl_points.joblib")

TRAIN_SEASONS = [
    2016, 2017, 2018, 2019, 2020, 2021,
    2022, 2023, 2024, 2025, 2026
]

MIN_PRIOR_GAMES = 5

FEATURES = [
    "home",
    "player_games_played",
    "points_last_5",
    "points_last_10",
    "points_season",
    "goals_last_5",
    "goals_last_10",
    "goals_season",
    "assists_last_5",
    "assists_last_10",
    "assists_season",
    "shots_on_goal_last_5",
    "shots_on_goal_last_10",
    "shots_on_goal_season",
    "shots_missed_last_5",
    "shots_missed_last_10",
    "shots_missed_season",
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


def main():
    data = pd.read_csv(INPUT_FILE)

    train = data[
        data["season"].isin(TRAIN_SEASONS)
        & (data["player_games_played"] >= MIN_PRIOR_GAMES)
    ].copy()

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
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
        train[FEATURES],
        train["points"]
    )

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "train_seasons": TRAIN_SEASONS,
            "min_prior_games": MIN_PRIOR_GAMES
        },
        MODEL_FILE
    )

    print("NHL Points Production Model")
    print("===========================")
    print(f"Training rows: {len(train)}")
    print(f"Features: {len(FEATURES)}")
    print(f"Saved: {MODEL_FILE}")


if __name__ == "__main__":
    main()