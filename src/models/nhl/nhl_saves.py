from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


INPUT_FILE = Path("data/processed/nhl_goalie_features.csv")
MODEL_FILE = Path("models/nhl_saves.joblib")

TRAIN_SEASONS = [
    2016, 2017, 2018, 2019, 2020, 2021,
    2022, 2023, 2024, 2025, 2026
]

MIN_PRIOR_GAMES = 5
MIN_TARGET_TOI = 2700

FEATURES = [
    "home",
    "goalie_games_played",
    "saves_last_5",
    "saves_last_10",
    "saves_season",
    "shots_against_last_5",
    "shots_against_last_10",
    "shots_against_season",
    "goals_against_last_5",
    "goals_against_last_10",
    "goals_against_season",
    "save_pct_last_5",
    "save_pct_last_10",
    "save_pct_season",
    "even_strength_saves_last_5",
    "even_strength_saves_last_10",
    "even_strength_saves_season",
    "power_play_saves_last_5",
    "power_play_saves_last_10",
    "power_play_saves_season",
    "shorthanded_saves_last_5",
    "shorthanded_saves_last_10",
    "shorthanded_saves_season",
    "time_on_ice_last_5",
    "time_on_ice_last_10",
    "time_on_ice_season",
    "saves_per_60_last_5",
    "shots_against_per_60_last_5",
    "days_rest",
    "back_to_back",
    "opponent_shots_for_last_5",
    "opponent_shots_for_season",
    "opponent_goals_for_last_5",
    "opponent_goals_for_season",
    "opponent_shooting_pct_last_5",
    "opponent_shooting_pct_season",
    "opponent_power_play_pct_last_5",
    "opponent_power_play_pct_season",
    "opponent_days_rest",
    "opponent_back_to_back"
]


def main():
    data = pd.read_csv(INPUT_FILE)

    train = data[
        data["season"].isin(TRAIN_SEASONS)
        & (data["goalie_games_played"] >= MIN_PRIOR_GAMES)
        & (data["time_on_ice"] >= MIN_TARGET_TOI)
    ].copy()

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            RandomForestRegressor(
                n_estimators=500,
                max_depth=7,
                min_samples_leaf=8,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    model.fit(
        train[FEATURES],
        train["saves"]
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
            "min_prior_games": MIN_PRIOR_GAMES,
            "min_target_toi": MIN_TARGET_TOI
        },
        MODEL_FILE
    )

    print("NHL Saves Production Model")
    print("==========================")
    print(f"Training rows: {len(train)}")
    print(f"Features: {len(FEATURES)}")
    print(f"Saved: {MODEL_FILE}")


if __name__ == "__main__":
    main()