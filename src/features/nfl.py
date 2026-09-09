import os
import pandas as pd

INPUT_FILE = "data/processed/nfl_games.csv"
OUTPUT_FILE = "data/processed/nfl_features.csv"

df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

team_games = {}

def get_previous_games(team):
    return team_games.get(team, [])

def average(games, field):
    values = [game[field] for game in games[-5:] if pd.notna(game[field])]
    return sum(values) / len(values) if values else None

def location_games(games, location):
    return [game for game in games if game["location"] == location]

rows = []

for _, game in df.iterrows():
    home = game["home_team"]
    away = game["away_team"]
    home_previous = get_previous_games(home)
    away_previous = get_previous_games(away)
    home_at_home = location_games(home_previous, "home")
    away_on_road = location_games(away_previous, "away")
    home_wins = sum(game["result"] for game in home_previous[-5:])
    away_wins = sum(game["result"] for game in away_previous[-5:])

    if game["home_score"] > game["away_score"]:
        home_result = 1.0
        away_result = 0.0
        home_win = 1
    elif game["away_score"] > game["home_score"]:
        home_result = 0.0
        away_result = 1.0
        home_win = 0
    else:
        home_result = 0.5
        away_result = 0.5
        home_win = None

    rows.append({
        "game_id": game["game_id"],
        "date": game["date"],
        "season": game["season"],
        "week": game["week"],
        "home_team": home,
        "away_team": away,
        "home_avg_points_last_5": average(home_previous, "points"),
        "away_avg_points_last_5": average(away_previous, "points"),
        "home_avg_points_allowed_last_5": average(home_previous, "points_allowed"),
        "away_avg_points_allowed_last_5": average(away_previous, "points_allowed"),
        "home_avg_point_diff_last_5": average(home_previous, "point_diff"),
        "away_avg_point_diff_last_5": average(away_previous, "point_diff"),
        "home_avg_yards_last_5": average(home_previous, "yards"),
        "away_avg_yards_last_5": average(away_previous, "yards"),
        "home_avg_yards_allowed_last_5": average(home_previous, "yards_allowed"),
        "away_avg_yards_allowed_last_5": average(away_previous, "yards_allowed"),
        "home_avg_passing_yards_last_5": average(home_previous, "passing_yards"),
        "away_avg_passing_yards_last_5": average(away_previous, "passing_yards"),
        "home_avg_rushing_yards_last_5": average(home_previous, "rushing_yards"),
        "away_avg_rushing_yards_last_5": average(away_previous, "rushing_yards"),
        "home_avg_turnovers_last_5": average(home_previous, "turnovers"),
        "away_avg_turnovers_last_5": average(away_previous, "turnovers"),
        "home_avg_turnover_diff_last_5": average(home_previous, "turnover_diff"),
        "away_avg_turnover_diff_last_5": average(away_previous, "turnover_diff"),
        "home_avg_first_downs_last_5": average(home_previous, "first_downs"),
        "away_avg_first_downs_last_5": average(away_previous, "first_downs"),
        "home_avg_third_down_pct_last_5": average(home_previous, "third_down_pct"),
        "away_avg_third_down_pct_last_5": average(away_previous, "third_down_pct"),
        "home_avg_sacks_allowed_last_5": average(home_previous, "sacks_allowed"),
        "away_avg_sacks_allowed_last_5": average(away_previous, "sacks_allowed"),
        "home_avg_sacks_generated_last_5": average(home_previous, "sacks_generated"),
        "away_avg_sacks_generated_last_5": average(away_previous, "sacks_generated"),
        "home_win_pct_last_5": home_wins / len(home_previous[-5:]) if home_previous else None,
        "away_win_pct_last_5": away_wins / len(away_previous[-5:]) if away_previous else None,
        "home_avg_points_at_home_last_5": average(home_at_home, "points"),
        "away_avg_points_on_road_last_5": average(away_on_road, "points"),
        "home_win": home_win
    })

    home_turnovers = float(game["home_turnovers"])
    away_turnovers = float(game["away_turnovers"])

    team_games.setdefault(home, []).append({
        "points": game["home_score"],
        "points_allowed": game["away_score"],
        "point_diff": game["home_score"] - game["away_score"],
        "yards": float(game["home_total_yards"]),
        "yards_allowed": float(game["away_total_yards"]),
        "passing_yards": float(game["home_passing_yards"]),
        "rushing_yards": float(game["home_rushing_yards"]),
        "turnovers": home_turnovers,
        "turnover_diff": away_turnovers - home_turnovers,
        "first_downs": float(game["home_first_downs"]),
        "third_down_pct": game["home_third_down_pct"],
        "sacks_allowed": game["home_sacks_allowed"],
        "sacks_generated": game["away_sacks_allowed"],
        "result": home_result,
        "location": "home"
    })
    team_games.setdefault(away, []).append({
        "points": game["away_score"],
        "points_allowed": game["home_score"],
        "point_diff": game["away_score"] - game["home_score"],
        "yards": float(game["away_total_yards"]),
        "yards_allowed": float(game["home_total_yards"]),
        "passing_yards": float(game["away_passing_yards"]),
        "rushing_yards": float(game["away_rushing_yards"]),
        "turnovers": away_turnovers,
        "turnover_diff": home_turnovers - away_turnovers,
        "first_downs": float(game["away_first_downs"]),
        "third_down_pct": game["away_third_down_pct"],
        "sacks_allowed": game["away_sacks_allowed"],
        "sacks_generated": game["home_sacks_allowed"],
        "result": away_result,
        "location": "away"
    })

features = pd.DataFrame(rows)
os.makedirs("data/processed", exist_ok=True)
features.to_csv(OUTPUT_FILE, index=False)

print(features)
print(f"Done. Created features for {len(features)} games.")
print(f"Ties excluded from winner target: {features['home_win'].isna().sum()}")