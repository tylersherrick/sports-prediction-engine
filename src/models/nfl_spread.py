import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

GAMES_FILE = "data/processed/nfl_games.csv"
FEATURES_FILE = "data/processed/nfl_features.csv"

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
df["home_margin"] = df["home_score"] - df["away_score"]
df = df.dropna(subset=features + ["home_margin"])

train = df[df["season"].isin([2022, 2023, 2024])]
test = df[df["season"] == 2025]

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)

model.fit(train[features], train["home_margin"])

predictions = model.predict(test[features])
mae = mean_absolute_error(test["home_margin"], predictions)

results = test[["away_team", "home_team", "home_margin"]].copy()
results["predicted_margin"] = predictions

winner_correct = (
    (results["predicted_margin"] > 0) ==
    (results["home_margin"] > 0)
)

print(f"Training games: {len(train)}")
print(f"Testing games: {len(test)}")
print(f"Mean absolute error: {mae:.2f} points")
print(f"Winner from margin: {winner_correct.sum()}/{len(results)} ({winner_correct.mean():.1%})")