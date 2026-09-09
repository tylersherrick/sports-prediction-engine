import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

GAMES_FILE = "data/processed/nfl_games.csv"
FEATURES_FILE = "data/processed/nfl_features.csv"
OUTPUT_FILE = "models/nfl_2026_total.joblib"

features = [
    "home_avg_points_last_5",
    "away_avg_points_last_5",
    "home_avg_points_allowed_last_5",
    "away_avg_points_allowed_last_5",
    "home_avg_yards_last_5",
    "away_avg_yards_last_5",
    "home_avg_yards_allowed_last_5",
    "away_avg_yards_allowed_last_5",
    "home_avg_passing_yards_last_5",
    "away_avg_passing_yards_last_5",
    "home_avg_turnovers_last_5",
    "away_avg_turnovers_last_5",
    "home_avg_turnover_diff_last_5",
    "away_avg_turnover_diff_last_5",
    "home_win_pct_last_5",
    "away_win_pct_last_5",
    "home_yards_matchup",
    "away_yards_matchup",
    "home_points_matchup",
    "away_points_matchup"
]

df = pd.read_csv(FEATURES_FILE)
games = pd.read_csv(GAMES_FILE)[["game_id", "home_score", "away_score"]]

df["home_yards_matchup"] = (
    df["home_avg_yards_last_5"] - df["away_avg_yards_allowed_last_5"]
)
df["away_yards_matchup"] = (
    df["away_avg_yards_last_5"] - df["home_avg_yards_allowed_last_5"]
)
df["home_points_matchup"] = (
    df["home_avg_points_last_5"] - df["away_avg_points_allowed_last_5"]
)
df["away_points_matchup"] = (
    df["away_avg_points_last_5"] - df["home_avg_points_allowed_last_5"]
)

df = df.merge(games, on="game_id")
df["total_points"] = df["home_score"] + df["away_score"]

train = df[
    df["season"].isin([2022, 2023, 2024, 2025])
].dropna(subset=features + ["total_points"])

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)

model.fit(train[features], train["total_points"])

os.makedirs("models", exist_ok=True)

joblib.dump({
    "model": model,
    "features": features
}, OUTPUT_FILE)

print(f"Training games: {len(train)}")
print("Training seasons: 2022-2025")
print(f"Model saved: {OUTPUT_FILE}")