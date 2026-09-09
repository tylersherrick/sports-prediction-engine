import joblib
import pandas as pd

MODEL_FILE = "models/nfl_2026_predictor.joblib"
DATA_FILE = "data/processed/nfl_games.csv"

AWAY_TEAM = "New England Patriots"
HOME_TEAM = "Seattle Seahawks"

saved = joblib.load(MODEL_FILE)
model = saved["model"]
features = saved["features"]

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

def team_history(team):
    games = df[
        (df["home_team"] == team) |
        (df["away_team"] == team)
    ].tail(5)

    history = []

    for _, game in games.iterrows():
        is_home = game["home_team"] == team

        if is_home:
            points = game["home_score"]
            points_allowed = game["away_score"]
            yards = game["home_total_yards"]
            yards_allowed = game["away_total_yards"]
            passing_yards = game["home_passing_yards"]
            turnovers = game["home_turnovers"]
            opponent_turnovers = game["away_turnovers"]
        else:
            points = game["away_score"]
            points_allowed = game["home_score"]
            yards = game["away_total_yards"]
            yards_allowed = game["home_total_yards"]
            passing_yards = game["away_passing_yards"]
            turnovers = game["away_turnovers"]
            opponent_turnovers = game["home_turnovers"]

        if points > points_allowed:
            result = 1
        elif points < points_allowed:
            result = 0
        else:
            result = 0.5

        history.append({
            "points": float(points),
            "points_allowed": float(points_allowed),
            "yards": float(yards),
            "yards_allowed": float(yards_allowed),
            "passing_yards": float(passing_yards),
            "turnovers": float(turnovers),
            "turnover_diff": float(opponent_turnovers) - float(turnovers),
            "result": result
        })

    return history

def average(history, field):
    return sum(game[field] for game in history) / len(history)

def american_odds(probability):
    if probability >= 0.5:
        return round(-100 * probability / (1 - probability))
    return round(100 * (1 - probability) / probability)

def format_odds(odds):
    return f"+{odds}" if odds > 0 else str(odds)

home = team_history(HOME_TEAM)
away = team_history(AWAY_TEAM)

row = {
    "home_avg_points_last_5": average(home, "points"),
    "away_avg_points_last_5": average(away, "points"),
    "home_avg_points_allowed_last_5": average(home, "points_allowed"),
    "away_avg_points_allowed_last_5": average(away, "points_allowed"),
    "home_avg_yards_last_5": average(home, "yards"),
    "away_avg_yards_last_5": average(away, "yards"),
    "home_avg_yards_allowed_last_5": average(home, "yards_allowed"),
    "away_avg_yards_allowed_last_5": average(away, "yards_allowed"),
    "home_avg_passing_yards_last_5": average(home, "passing_yards"),
    "away_avg_passing_yards_last_5": average(away, "passing_yards"),
    "home_avg_turnovers_last_5": average(home, "turnovers"),
    "away_avg_turnovers_last_5": average(away, "turnovers"),
    "home_avg_turnover_diff_last_5": average(home, "turnover_diff"),
    "away_avg_turnover_diff_last_5": average(away, "turnover_diff"),
    "home_win_pct_last_5": average(home, "result"),
    "away_win_pct_last_5": average(away, "result")
}

row["home_yards_matchup"] = (
    row["home_avg_yards_last_5"] -
    row["away_avg_yards_allowed_last_5"]
)

row["away_yards_matchup"] = (
    row["away_avg_yards_last_5"] -
    row["home_avg_yards_allowed_last_5"]
)

row["home_points_matchup"] = (
    row["home_avg_points_last_5"] -
    row["away_avg_points_allowed_last_5"]
)

row["away_points_matchup"] = (
    row["away_avg_points_last_5"] -
    row["home_avg_points_allowed_last_5"]
)

X = pd.DataFrame([row])[features]

home_probability = model.predict_proba(X)[0][1]
away_probability = 1 - home_probability

home_odds = american_odds(home_probability)
away_odds = american_odds(away_probability)

pick = HOME_TEAM if home_probability >= 0.5 else AWAY_TEAM

print(f"\n{AWAY_TEAM} @ {HOME_TEAM}")
print(f"Pick: {pick}")
print(f"{HOME_TEAM}: {home_probability:.1%} ({format_odds(home_odds)})")
print(f"{AWAY_TEAM}: {away_probability:.1%} ({format_odds(away_odds)})")