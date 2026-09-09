import pandas as pd
from sklearn.ensemble import RandomForestRegressor

FEATURES_FILE = "data/processed/nfl_features.csv"
GAMES_FILE = "data/processed/nfl_games.csv"
ODDS_FILE = "data/betting/nfl_2025_odds.csv"

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

team_map = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS"
}

df = pd.read_csv(FEATURES_FILE)
games = pd.read_csv(GAMES_FILE)[["game_id", "home_score", "away_score"]]
odds = pd.read_csv(ODDS_FILE)

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
df = df.dropna(subset=features + ["total_points"])

train = df[df["season"].isin([2022, 2023, 2024])]
test = df[df["season"] == 2025].copy()

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)

model.fit(train[features], train["total_points"])
test["predicted_total"] = model.predict(test[features])

test["home_abbr"] = test["home_team"].map(team_map)
test["away_abbr"] = test["away_team"].map(team_map)

results = test.merge(
    odds,
    left_on=["week", "home_abbr", "away_abbr"],
    right_on=["week", "home_team", "away_team"],
    how="inner"
)

results["model_edge"] = results["predicted_total"] - results["total_line"]
results["model_pick"] = results["model_edge"].apply(
    lambda x: "over" if x > 0 else "under"
)

results = results[results["total_points"] != results["total_line"]].copy()

results["correct"] = (
    ((results["model_pick"] == "over") & (results["total_points"] > results["total_line"])) |
    ((results["model_pick"] == "under") & (results["total_points"] < results["total_line"]))
)

print(f"Games matched: {len(results)}")
print(f"O/U picks correct: {results['correct'].sum()}/{len(results)}")
print(f"O/U accuracy: {results['correct'].mean():.1%}")

for edge in [1, 2, 3, 4, 5]:
    subset = results[results["model_edge"].abs() >= edge]

    if len(subset):
        print(
            f"{edge}+ point edge: "
            f"{subset['correct'].sum()}/{len(subset)} "
            f"({subset['correct'].mean():.1%})"
        )