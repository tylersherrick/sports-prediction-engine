import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURES_FILE = "data/processed/nfl_features.csv"
ODDS_FILE = "data/betting/nfl_2025_odds.csv"

TEAM_CODES = {
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

df = df.dropna(subset=features + ["home_win"])

train = df[df["season"].isin([2022, 2023, 2024])]
test = df[df["season"] == 2025].copy()

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)

model.fit(train[features], train["home_win"])

test["predicted_home_win"] = model.predict(test[features])

test["home_code"] = test["home_team"].map(TEAM_CODES)
test["away_code"] = test["away_team"].map(TEAM_CODES)

test["predicted_team"] = test.apply(
    lambda game: game["home_code"]
    if game["predicted_home_win"] == 1
    else game["away_code"],
    axis=1
)

test["actual_winner"] = test.apply(
    lambda game: game["home_code"]
    if game["home_win"] == 1
    else game["away_code"],
    axis=1
)

odds = pd.read_csv(ODDS_FILE)

odds["favorite"] = odds.apply(
    lambda game: game["home_team"]
    if game["home_moneyline"] < game["away_moneyline"]
    else game["away_team"],
    axis=1
)

odds["underdog"] = odds.apply(
    lambda game: game["away_team"]
    if game["home_moneyline"] < game["away_moneyline"]
    else game["home_team"],
    axis=1
)

results = test.merge(
    odds,
    left_on=["week", "home_code", "away_code"],
    right_on=["week", "home_team", "away_team"],
    how="inner"
)

results["correct"] = results["predicted_team"] == results["actual_winner"]
results["picked_underdog"] = results["predicted_team"] == results["underdog"]

underdog_picks = results[results["picked_underdog"]]
underdog_wins = underdog_picks["correct"].sum()

print(f"Games matched: {len(results)}")
print(f"Overall: {results['correct'].sum()}/{len(results)} ({results['correct'].mean():.1%})")
print()
print(f"Underdogs predicted: {len(underdog_picks)}")
print(f"Underdogs correct: {underdog_wins}")
print(f"Underdog accuracy: {underdog_picks['correct'].mean():.1%}")