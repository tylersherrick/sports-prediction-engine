from pathlib import Path

import pandas as pd

GOALIES_FILE = Path(
    "data/processed/nhl_goalies.csv"
)

GAMES_FILE = Path(
    "data/processed/nhl_features.csv"
)

OUTPUT_FILE = Path(
    "data/processed/nhl_goalie_features.csv"
)

ROLLING_WINDOWS = [
    5,
    10
]

GOALIE_STATS = [
    "goals_against",
    "shots_against",
    "saves",
    "save_pct",
    "even_strength_saves",
    "power_play_saves",
    "shorthanded_saves",
    "time_on_ice"
]

def add_goalie_features(
    goalies
):
    goalies = goalies.sort_values(
        [
            "player_id",
            "date",
            "game_id"
        ]
    ).reset_index(
        drop=True
    )

    goalies[
        "goalie_games_played"
    ] = (
        goalies
        .groupby(
            [
                "player_id",
                "season"
            ]
        )
        .cumcount()
    )

    for stat in GOALIE_STATS:
        for window in ROLLING_WINDOWS:
            goalies[
                f"{stat}_last_{window}"
            ] = (
                goalies
                .groupby(
                    "player_id"
                )[stat]
                .transform(
                    lambda values:
                    values
                    .shift(1)
                    .rolling(
                        window,
                        min_periods=1
                    )
                    .mean()
                )
            )

        goalies[
            f"{stat}_season"
        ] = (
            goalies
            .groupby(
                [
                    "player_id",
                    "season"
                ]
            )[stat]
            .transform(
                lambda values:
                values
                .shift(1)
                .expanding(
                    min_periods=1
                )
                .mean()
            )
        )

    previous_date = (
        goalies
        .groupby(
            "player_id"
        )["date"]
        .shift(1)
    )

    goalies[
        "days_since_last_game"
    ] = (
        goalies["date"]
        - previous_date
    ).dt.total_seconds() / 86400

    goalies[
        "days_rest"
    ] = (
        goalies[
            "days_since_last_game"
        ]
        .sub(1)
        .clip(
            lower=0,
            upper=10
        )
    )

    goalies[
        "back_to_back"
    ] = (
        goalies[
            "days_since_last_game"
        ]
        .le(1.5)
        .fillna(False)
        .astype(int)
    )

    return goalies

def build_opponent_features(
    games
):
    home = games[
        [
            "game_id",
            "home_team_id",
            "home_shots_for_last_5",
            "home_shots_for_season",
            "home_goals_for_last_5",
            "home_goals_for_season",
            "home_shooting_pct_last_5",
            "home_shooting_pct_season",
            "home_power_play_pct_last_5",
            "home_power_play_pct_season",
            "home_days_rest",
            "home_back_to_back"
        ]
    ].copy()

    home = home.rename(
        columns={
            "home_team_id":
                "opponent_team_id",
            "home_shots_for_last_5":
                "opponent_shots_for_last_5",
            "home_shots_for_season":
                "opponent_shots_for_season",
            "home_goals_for_last_5":
                "opponent_goals_for_last_5",
            "home_goals_for_season":
                "opponent_goals_for_season",
            "home_shooting_pct_last_5":
                "opponent_shooting_pct_last_5",
            "home_shooting_pct_season":
                "opponent_shooting_pct_season",
            "home_power_play_pct_last_5":
                "opponent_power_play_pct_last_5",
            "home_power_play_pct_season":
                "opponent_power_play_pct_season",
            "home_days_rest":
                "opponent_days_rest",
            "home_back_to_back":
                "opponent_back_to_back"
        }
    )

    away = games[
        [
            "game_id",
            "away_team_id",
            "away_shots_for_last_5",
            "away_shots_for_season",
            "away_goals_for_last_5",
            "away_goals_for_season",
            "away_shooting_pct_last_5",
            "away_shooting_pct_season",
            "away_power_play_pct_last_5",
            "away_power_play_pct_season",
            "away_days_rest",
            "away_back_to_back"
        ]
    ].copy()

    away = away.rename(
        columns={
            "away_team_id":
                "opponent_team_id",
            "away_shots_for_last_5":
                "opponent_shots_for_last_5",
            "away_shots_for_season":
                "opponent_shots_for_season",
            "away_goals_for_last_5":
                "opponent_goals_for_last_5",
            "away_goals_for_season":
                "opponent_goals_for_season",
            "away_shooting_pct_last_5":
                "opponent_shooting_pct_last_5",
            "away_shooting_pct_season":
                "opponent_shooting_pct_season",
            "away_power_play_pct_last_5":
                "opponent_power_play_pct_last_5",
            "away_power_play_pct_season":
                "opponent_power_play_pct_season",
            "away_days_rest":
                "opponent_days_rest",
            "away_back_to_back":
                "opponent_back_to_back"
        }
    )

    return pd.concat(
        [
            home,
            away
        ],
        ignore_index=True
    )

def main():
    goalies = pd.read_csv(
        GOALIES_FILE
    )

    games = pd.read_csv(
        GAMES_FILE
    )

    goalies[
        "game_id"
    ] = (
        goalies[
            "game_id"
        ]
        .astype(str)
    )

    games[
        "game_id"
    ] = (
        games[
            "game_id"
        ]
        .astype(str)
    )

    goalies[
        "player_id"
    ] = (
        goalies[
            "player_id"
        ]
        .astype(str)
    )

    goalies[
        "team_id"
    ] = (
        goalies[
            "team_id"
        ]
        .astype(str)
    )

    goalies[
        "opponent_team_id"
    ] = (
        goalies[
            "opponent_team_id"
        ]
        .astype(str)
    )

    games[
        "home_team_id"
    ] = (
        games[
            "home_team_id"
        ]
        .astype(str)
    )

    games[
        "away_team_id"
    ] = (
        games[
            "away_team_id"
        ]
        .astype(str)
    )

    goalies[
        "date"
    ] = pd.to_datetime(
        goalies[
            "date"
        ],
        utc=True
    )

    print(
        f"Loaded "
        f"{len(goalies)} "
        f"goalie-game rows."
    )

    goalies = add_goalie_features(
        goalies
    )

    opponent_features = (
        build_opponent_features(
            games
        )
    )

    features = goalies.merge(
        opponent_features,
        on=[
            "game_id",
            "opponent_team_id"
        ],
        how="left",
        validate="many_to_one"
    )

    features[
        "toi_minutes_last_5"
    ] = (
        features[
            "time_on_ice_last_5"
        ]
        / 60
    )

    features[
        "saves_per_60_last_5"
    ] = (
        features[
            "saves_last_5"
        ]
        / features[
            "toi_minutes_last_5"
        ]
        * 60
    )

    features[
        "shots_against_per_60_last_5"
    ] = (
        features[
            "shots_against_last_5"
        ]
        / features[
            "toi_minutes_last_5"
        ]
        * 60
    )

    features = features.sort_values(
        [
            "date",
            "game_id",
            "team",
            "player"
        ]
    ).reset_index(
        drop=True
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    features.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Feature rows: "
        f"{len(features)}"
    )

    print(
        f"Feature columns: "
        f"{len(features.columns)}"
    )

    print(
        f"Unique goalies: "
        f"{features['player_id'].nunique()}"
    )

    print(
        f"Saved "
        f"{OUTPUT_FILE}"
    )

    print(
        "\nSaves target:"
    )

    print(
        features[
            "saves"
        ]
        .describe()
    )

    print(
        "\nPregame feature sample:"
    )

    sample_columns = [
        "date",
        "player",
        "team",
        "opponent",
        "goalie_games_played",
        "saves",
        "saves_last_5",
        "saves_last_10",
        "saves_season",
        "shots_against_last_5",
        "save_pct_last_5",
        "time_on_ice_last_5",
        "saves_per_60_last_5",
        "opponent_shots_for_last_5"
    ]

    print(
        features[
            sample_columns
        ]
        .tail(20)
        .to_string(
            index=False
        )
    )

if __name__ == "__main__":
    main()