import os
import pandas as pd

URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
OUTPUT_FILE = "data/betting/nfl_2025_odds.csv"

df = pd.read_csv(URL)

df = df[
    (df["season"] == 2025) &
    (df["game_type"] == "REG")
].copy()

columns = [
    "game_id",
    "week",
    "away_team",
    "home_team",
    "away_moneyline",
    "home_moneyline",
    "spread_line",
    "away_spread_odds",
    "home_spread_odds",
    "total_line"
]

df = df[columns]

os.makedirs("data/betting", exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(df)} games.")
print(df.columns.tolist())