from pathlib import Path

import pandas as pd

SKATERS_FILE = Path(
    "data/processed/nhl_skaters.csv"
)

GAMES_FILE = Path(
    "data/processed/nhl_features.csv"
)

OUTPUT_FILE = Path(
    "data/processed/nhl_skater_features.csv"
)

ROLLING_WINDOWS = [
    5,
    10
]

PLAYER_STATS = [
    "shots_on_goal",
    "shots_missed",
    "goals",
    "assists",
    "points",
    "time_on_ice",
    "power_play_time_on_ice",
    "even_strength_time_on_ice",
    "shifts"
]

def add_player_features(
    skaters
):
    skaters = skaters.sort_values(
        [
            "player_id",
            "date",
            "game_id"
        ]
    ).reset_index(
        drop=True
    )

    skaters[
        "player_games_played"
    ] = (
        skaters
        .groupby(
            [
                "player_id",
                "season"
            ]
        )
        .cumcount()
    )

    for stat in PLAYER_STATS:
        for window in ROLLING_WINDOWS:
            skaters[
                f"{stat}_last_{window}"
            ] = (
                skaters
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

        skaters[
            f"{stat}_season"
        ] = (
            skaters
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
        skaters
        .groupby(
            "player_id"
        )["date"]
        .shift(1)
    )

    skaters[
        "days_since_last_game"
    ] = (
        skaters["date"]
        - previous_date
    ).dt.total_seconds() / 86400

    skaters[
        "days_rest"
    ] = (
        skaters[
            "days_since_last_game"
        ]
        .sub(1)
        .clip(
            lower=0,
            upper=10
        )
    )

    skaters[
        "back_to_back"
    ] = (
        skaters[
            "days_since_last_game"
        ]
        .le(1.5)
        .fillna(False)
        .astype(int)
    )

    return skaters

def build_opponent_features(
    games
):
    home = games[
        [
            "game_id",
            "home_team_id",
            "home_shots_against_last_5",
            "home_shots_against_season",
            "home_goals_against_last_5",
            "home_goals_against_season",
            "home_save_pct_last_5",
            "home_save_pct_season",
            "home_penalty_minutes_last_5",
            "home_penalty_minutes_season",
            "home_days_rest",
            "home_back_to_back"
        ]
    ].copy()

    home = home.rename(
        columns={
            "home_team_id":
                "opponent_team_id",
            "home_shots_against_last_5":
                "opponent_shots_against_last_5",
            "home_shots_against_season":
                "opponent_shots_against_season",
            "home_goals_against_last_5":
                "opponent_goals_against_last_5",
            "home_goals_against_season":
                "opponent_goals_against_season",
            "home_save_pct_last_5":
                "opponent_save_pct_last_5",
            "home_save_pct_season":
                "opponent_save_pct_season",
            "home_penalty_minutes_last_5":
                "opponent_penalty_minutes_last_5",
            "home_penalty_minutes_season":
                "opponent_penalty_minutes_season",
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
            "away_shots_against_last_5",
            "away_shots_against_season",
            "away_goals_against_last_5",
            "away_goals_against_season",
            "away_save_pct_last_5",
            "away_save_pct_season",
            "away_penalty_minutes_last_5",
            "away_penalty_minutes_season",
            "away_days_rest",
            "away_back_to_back"
        ]
    ].copy()

    away = away.rename(
        columns={
            "away_team_id":
                "opponent_team_id",
            "away_shots_against_last_5":
                "opponent_shots_against_last_5",
            "away_shots_against_season":
                "opponent_shots_against_season",
            "away_goals_against_last_5":
                "opponent_goals_against_last_5",
            "away_goals_against_season":
                "opponent_goals_against_season",
            "away_save_pct_last_5":
                "opponent_save_pct_last_5",
            "away_save_pct_season":
                "opponent_save_pct_season",
            "away_penalty_minutes_last_5":
                "opponent_penalty_minutes_last_5",
            "away_penalty_minutes_season":
                "opponent_penalty_minutes_season",
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
    skaters = pd.read_csv(
        SKATERS_FILE
    )

    games = pd.read_csv(
        GAMES_FILE
    )

    skaters[
        "game_id"
    ] = (
        skaters[
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

    skaters[
        "player_id"
    ] = (
        skaters[
            "player_id"
        ]
        .astype(str)
    )

    skaters[
        "team_id"
    ] = (
        skaters[
            "team_id"
        ]
        .astype(str)
    )

    skaters[
        "opponent_team_id"
    ] = (
        skaters[
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

    skaters[
        "date"
    ] = pd.to_datetime(
        skaters[
            "date"
        ],
        utc=True
    )

    print(
        f"Loaded "
        f"{len(skaters)} "
        f"skater-game rows."
    )

    skaters = add_player_features(
        skaters
    )

    opponent_features = (
        build_opponent_features(
            games
        )
    )

    features = skaters.merge(
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
        "pp_toi_minutes_last_5"
    ] = (
        features[
            "power_play_time_on_ice_last_5"
        ]
        / 60
    )

    features[
        "shots_per_60_last_5"
    ] = (
        features[
            "shots_on_goal_last_5"
        ]
        / features[
            "toi_minutes_last_5"
        ]
        * 60
    )

    features[
        "pp_usage_share_last_5"
    ] = (
        features[
            "power_play_time_on_ice_last_5"
        ]
        / features[
            "time_on_ice_last_5"
        ]
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
        f"Unique players: "
        f"{features['player_id'].nunique()}"
    )

    print(
        f"Saved "
        f"{OUTPUT_FILE}"
    )

    print(
        "\nSOG target:"
    )

    print(
        features[
            "shots_on_goal"
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
        "player_games_played",
        "shots_on_goal",
        "shots_on_goal_last_5",
        "shots_on_goal_last_10",
        "shots_on_goal_season",
        "shots_missed_last_5",
        "time_on_ice_last_5",
        "power_play_time_on_ice_last_5",
        "shots_per_60_last_5",
        "opponent_shots_against_last_5"
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