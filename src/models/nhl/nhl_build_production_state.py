import os
import pandas as pd


TEAM_FILE = "data/processed/nhl_features.csv"
SKATER_FILE = "data/processed/nhl_skater_features.csv"
GOALIE_FILE = "data/processed/nhl_goalie_features.csv"

OUTPUT_DIR = "models"

TEAM_OUTPUT = f"{OUTPUT_DIR}/nhl_team_state.csv"
SKATER_OUTPUT = f"{OUTPUT_DIR}/nhl_skater_state.csv"
GOALIE_OUTPUT = f"{OUTPUT_DIR}/nhl_goalie_state.csv"

PLAYER_RECENCY_DAYS = 120


def build_team_state():
    df = pd.read_csv(
        TEAM_FILE,
        dtype={
            "game_id": str,
            "home_team_id": str,
            "away_team_id": str
        }
    )

    df["date"] = pd.to_datetime(
        df["date"],
        utc=True
    )

    rows = []

    team_ids = set(
        df["home_team_id"]
    ) | set(
        df["away_team_id"]
    )

    for team_id in team_ids:
        home = df[
            df["home_team_id"] == team_id
        ]

        away = df[
            df["away_team_id"] == team_id
        ]

        games = pd.concat([
            home,
            away
        ]).sort_values("date")

        if not games.empty:
            rows.append(
                games.iloc[-1]
            )

    state = pd.DataFrame(rows)

    state.to_csv(
        TEAM_OUTPUT,
        index=False
    )

    print(
        f"Team state: {len(state)} rows"
    )


def build_player_state(
    input_file,
    output_file,
    label
):
    df = pd.read_csv(
        input_file,
        dtype={
            "game_id": str,
            "player_id": str,
            "team_id": str,
            "opponent_id": str
        }
    )

    df["date"] = pd.to_datetime(
        df["date"],
        utc=True
    )

    latest_team_dates = (
        df.groupby("team_id")["date"]
        .max()
        .to_dict()
    )

    latest_players = (
        df.sort_values("date")
        .groupby("player_id", as_index=False)
        .tail(1)
        .copy()
    )

    latest_players["team_latest_date"] = (
        latest_players["team_id"]
        .map(latest_team_dates)
    )

    latest_players = latest_players[
        latest_players["date"] >= (
            latest_players["team_latest_date"]
            - pd.Timedelta(
                days=PLAYER_RECENCY_DAYS
            )
        )
    ].copy()

    latest_players = latest_players.drop(
        columns=["team_latest_date"]
    )

    latest_players.to_csv(
        output_file,
        index=False
    )

    print(
        f"{label} state: "
        f"{len(latest_players)} rows"
    )


def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    build_team_state()

    build_player_state(
        SKATER_FILE,
        SKATER_OUTPUT,
        "Skater"
    )

    build_player_state(
        GOALIE_FILE,
        GOALIE_OUTPUT,
        "Goalie"
    )

    print()
    print("Production state created:")
    print(TEAM_OUTPUT)
    print(SKATER_OUTPUT)
    print(GOALIE_OUTPUT)


if __name__ == "__main__":
    main()