import glob
import json
import os
import pandas as pd

def get_stat(team, name):
    for stat in team["statistics"]:
        if stat["name"] == name:
            return stat["displayValue"]
    return None

def efficiency(value):
    if not value:
        return None

    made, attempts = value.split("-")
    return int(made) / int(attempts) if int(attempts) else 0

def sacks(value):
    if not value:
        return None

    return float(value.split("-")[0])

rows = []

for path in glob.glob("data/raw/nfl/*_details.json"):
    with open(path) as file:
        data = json.load(file)

    header = data["header"]["competitions"][0]
    teams = data["boxscore"]["teams"]
    home = next(team for team in teams if team["homeAway"] == "home")
    away = next(team for team in teams if team["homeAway"] == "away")
    competitors = header["competitors"]
    home_result = next(team for team in competitors if team["homeAway"] == "home")
    away_result = next(team for team in competitors if team["homeAway"] == "away")

    home_score = int(home_result["score"])
    away_score = int(away_result["score"])

    if home_score > away_score:
        winner = home["team"]["displayName"]
    elif away_score > home_score:
        winner = away["team"]["displayName"]
    else:
        winner = "TIE"

    rows.append({
        "game_id": header["id"],
        "date": header["date"],
        "season": data["header"]["season"]["year"],
        "week": data["header"]["week"],
        "home_team": home["team"]["displayName"],
        "away_team": away["team"]["displayName"],
        "home_score": home_score,
        "away_score": away_score,
        "winner": winner,
        "home_total_yards": get_stat(home, "totalYards"),
        "away_total_yards": get_stat(away, "totalYards"),
        "home_passing_yards": get_stat(home, "netPassingYards"),
        "away_passing_yards": get_stat(away, "netPassingYards"),
        "home_rushing_yards": get_stat(home, "rushingYards"),
        "away_rushing_yards": get_stat(away, "rushingYards"),
        "home_turnovers": get_stat(home, "turnovers"),
        "away_turnovers": get_stat(away, "turnovers"),
        "home_yards_per_play": get_stat(home, "yardsPerPlay"),
        "away_yards_per_play": get_stat(away, "yardsPerPlay"),
        "home_first_downs": get_stat(home, "firstDowns"),
        "away_first_downs": get_stat(away, "firstDowns"),
        "home_third_down_pct": efficiency(get_stat(home, "thirdDownEff")),
        "away_third_down_pct": efficiency(get_stat(away, "thirdDownEff")),
        "home_sacks_allowed": sacks(get_stat(home, "sacksYardsLost")),
        "away_sacks_allowed": sacks(get_stat(away, "sacksYardsLost"))
    })

os.makedirs("data/processed", exist_ok=True)

df = pd.DataFrame(rows)
df = df.sort_values("date")
df.to_csv("data/processed/nfl_games.csv", index=False)

print(df)
print(f"Done. Processed {len(df)} games.")
print(f"Ties found: {(df['winner'] == 'TIE').sum()}")