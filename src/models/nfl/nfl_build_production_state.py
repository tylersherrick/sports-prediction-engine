import os

import pandas as pd


GAMES_FILE = "data/processed/nfl_games.csv"
OUTPUT_FILE = "models/nfl_games_state.csv"

RECENT_GAMES_PER_TEAM = 5


def main():
    df = pd.read_csv(GAMES_FILE)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    teams = sorted(
        set(df["home_team"].dropna()) |
        set(df["away_team"].dropna())
    )

    indexes = set()

    for team in teams:
        team_games = df[
            (df["home_team"] == team) |
            (df["away_team"] == team)
        ].tail(RECENT_GAMES_PER_TEAM)

        indexes.update(team_games.index.tolist())

    state = df.loc[sorted(indexes)].copy()
    state = state.sort_values("date").reset_index(drop=True)

    os.makedirs("models", exist_ok=True)
    state.to_csv(OUTPUT_FILE, index=False)

    print("NFL Production State")
    print("====================")
    print(f"Teams: {len(teams)}")
    print(f"Games: {len(state)}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()