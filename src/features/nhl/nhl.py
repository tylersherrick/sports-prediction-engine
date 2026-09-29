from pathlib import Path

import numpy as np
import pandas as pd

INPUT_FILE = Path("data/processed/nhl_games.csv")
OUTPUT_FILE = Path("data/processed/nhl_features.csv")

ROLLING_WINDOWS = [3, 5, 7, 10, 15]


def safe_divide(numerator, denominator):
    return np.where(
        denominator > 0,
        numerator / denominator,
        np.nan
    )


def build_team_games(games):
    home = pd.DataFrame({
        "game_id": games["game_id"],
        "season": games["season"],
        "date": games["date"],
        "team_id": games["home_team_id"],
        "team": games["home_team"],
        "opponent_team_id": games["away_team_id"],
        "opponent": games["away_team"],
        "home": 1,
        "goals_for": games["home_score"],
        "goals_against": games["away_score"],
        "shots_for": games["home_shotsTotal"],
        "shots_against": games["away_shotsTotal"],
        "power_play_goals": games["home_powerPlayGoals"],
        "power_play_opportunities": games["home_powerPlayOpportunities"],
        "penalty_minutes": games["home_penaltyMinutes"],
        "win": games["home_win"]
    })

    away = pd.DataFrame({
        "game_id": games["game_id"],
        "season": games["season"],
        "date": games["date"],
        "team_id": games["away_team_id"],
        "team": games["away_team"],
        "opponent_team_id": games["home_team_id"],
        "opponent": games["home_team"],
        "home": 0,
        "goals_for": games["away_score"],
        "goals_against": games["home_score"],
        "shots_for": games["away_shotsTotal"],
        "shots_against": games["home_shotsTotal"],
        "power_play_goals": games["away_powerPlayGoals"],
        "power_play_opportunities": games["away_powerPlayOpportunities"],
        "penalty_minutes": games["away_penaltyMinutes"],
        "win": 1 - games["home_win"]
    })

    team_games = pd.concat(
        [home, away],
        ignore_index=True
    )

    team_games["date"] = pd.to_datetime(
        team_games["date"],
        utc=True
    )

    team_games = team_games.sort_values(
        ["team_id", "date", "game_id"]
    ).reset_index(drop=True)

    team_games["goal_diff"] = (
        team_games["goals_for"]
        - team_games["goals_against"]
    )

    team_games["shot_diff"] = (
        team_games["shots_for"]
        - team_games["shots_against"]
    )

    team_games["shooting_pct"] = safe_divide(
        team_games["goals_for"],
        team_games["shots_for"]
    )

    team_games["save_pct"] = (
        1
        - safe_divide(
            team_games["goals_against"],
            team_games["shots_against"]
        )
    )

    team_games["power_play_pct"] = safe_divide(
        team_games["power_play_goals"],
        team_games["power_play_opportunities"]
    )

    return team_games


def add_schedule_features(team_games):
    team_games = team_games.copy()

    schedule_data = {
        "games_last_4_days": [],
        "games_last_6_days": [],
        "three_in_four": [],
        "four_in_six": [],
        "road_streak": [],
        "home_streak": [],
        "road_trip_game": [],
        "home_after_road_trip": []
    }

    for _, group in team_games.groupby(
        ["team_id", "season"],
        sort=False
    ):
        group = group.sort_values(
            ["date", "game_id"]
        )

        dates = group["date"].tolist()
        home_values = group["home"].tolist()

        road_streak = 0
        home_streak = 0

        group_values = {
            key: []
            for key in schedule_data
        }

        for index, current_date in enumerate(dates):
            previous_dates = dates[:index]

            games_4 = sum(
                0 < (
                    current_date - previous_date
                ).total_seconds() / 86400 <= 4
                for previous_date in previous_dates
            )

            games_6 = sum(
                0 < (
                    current_date - previous_date
                ).total_seconds() / 86400 <= 6
                for previous_date in previous_dates
            )

            group_values[
                "games_last_4_days"
            ].append(games_4)

            group_values[
                "games_last_6_days"
            ].append(games_6)

            group_values[
                "three_in_four"
            ].append(
                int(games_4 >= 2)
            )

            group_values[
                "four_in_six"
            ].append(
                int(games_6 >= 3)
            )

            is_home = (
                home_values[index] == 1
            )

            if is_home:
                group_values[
                    "road_streak"
                ].append(0)

                group_values[
                    "home_streak"
                ].append(
                    home_streak + 1
                )

                group_values[
                    "road_trip_game"
                ].append(0)

                group_values[
                    "home_after_road_trip"
                ].append(
                    int(road_streak >= 2)
                )

                home_streak += 1
                road_streak = 0

            else:
                group_values[
                    "road_streak"
                ].append(
                    road_streak + 1
                )

                group_values[
                    "home_streak"
                ].append(0)

                group_values[
                    "road_trip_game"
                ].append(
                    road_streak + 1
                )

                group_values[
                    "home_after_road_trip"
                ].append(0)

                road_streak += 1
                home_streak = 0

        for column in schedule_data:
            values = pd.Series(
                group_values[column],
                index=group.index
            )

            schedule_data[column].append(
                values
            )

    for column, pieces in schedule_data.items():
        combined = pd.concat(
            pieces
        ).sort_index()

        team_games[column] = combined

    return team_games


def add_team_features(team_games):
    stats = [
        "goals_for",
        "goals_against",
        "goal_diff",
        "shots_for",
        "shots_against",
        "shot_diff",
        "shooting_pct",
        "save_pct",
        "power_play_pct",
        "penalty_minutes",
        "win"
    ]

    team_games["games_played"] = (
        team_games
        .groupby(
            ["team_id", "season"]
        )
        .cumcount()
    )

    for stat in stats:
        for window in ROLLING_WINDOWS:
            team_games[
                f"{stat}_last_{window}"
            ] = (
                team_games
                .groupby("team_id")[stat]
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

        team_games[
            f"{stat}_season"
        ] = (
            team_games
            .groupby(
                ["team_id", "season"]
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

    venue_stats = [
        "goals_for",
        "goals_against",
        "goal_diff",
        "shots_for",
        "shots_against",
        "shot_diff",
        "shooting_pct",
        "save_pct",
        "power_play_pct",
        "win"
    ]

    for stat in venue_stats:
        team_games[
            f"{stat}_venue_season"
        ] = (
            team_games
            .groupby(
                [
                    "team_id",
                    "season",
                    "home"
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
        team_games
        .groupby("team_id")["date"]
        .shift(1)
    )

    team_games["days_since_last_game"] = (
        team_games["date"]
        - previous_date
    ).dt.total_seconds() / 86400

    team_games["days_rest"] = (
        team_games[
            "days_since_last_game"
        ]
        .sub(1)
        .clip(
            lower=0,
            upper=10
        )
    )

    team_games["back_to_back"] = (
        team_games[
            "days_since_last_game"
        ]
        .le(1.5)
        .fillna(False)
        .astype(int)
    )

    team_games = add_schedule_features(
        team_games
    )

    return team_games


def get_feature_columns(team_games):
    excluded = {
        "game_id",
        "season",
        "date",
        "team_id",
        "team",
        "opponent_team_id",
        "opponent",
        "home",
        "goals_for",
        "goals_against",
        "shots_for",
        "shots_against",
        "power_play_goals",
        "power_play_opportunities",
        "penalty_minutes",
        "win",
        "goal_diff",
        "shot_diff",
        "shooting_pct",
        "save_pct",
        "power_play_pct",
        "days_since_last_game"
    }

    return [
        column
        for column in team_games.columns
        if column not in excluded
    ]


def build_game_features(
    games,
    team_games
):
    feature_columns = get_feature_columns(
        team_games
    )

    home_features = (
        team_games[
            team_games["home"] == 1
        ][
            ["game_id", "team_id"]
            + feature_columns
        ]
        .copy()
    )

    away_features = (
        team_games[
            team_games["home"] == 0
        ][
            ["game_id", "team_id"]
            + feature_columns
        ]
        .copy()
    )

    home_features = home_features.rename(
        columns={
            column: f"home_{column}"
            for column in feature_columns
        }
    )

    away_features = away_features.rename(
        columns={
            column: f"away_{column}"
            for column in feature_columns
        }
    )

    home_features = home_features.rename(
        columns={
            "team_id":
            "home_feature_team_id"
        }
    )

    away_features = away_features.rename(
        columns={
            "team_id":
            "away_feature_team_id"
        }
    )

    features = games[
        [
            "game_id",
            "season",
            "date",
            "home_team_id",
            "home_team",
            "home_abbreviation",
            "away_team_id",
            "away_team",
            "away_abbreviation",
            "home_score",
            "away_score",
            "total_goals",
            "home_goal_diff",
            "home_win"
        ]
    ].copy()

    features = features.merge(
        home_features,
        on="game_id",
        how="left"
    )

    features = features.merge(
        away_features,
        on="game_id",
        how="left"
    )

    matchup_stats = []

    base_matchup_stats = [
        "goals_for",
        "goals_against",
        "goal_diff",
        "shots_for",
        "shots_against",
        "shot_diff",
        "shooting_pct",
        "save_pct",
        "power_play_pct",
        "penalty_minutes",
        "win"
    ]

    for stat in base_matchup_stats:
        for window in ROLLING_WINDOWS:
            matchup_stats.append(
                f"{stat}_last_{window}"
            )

        matchup_stats.append(
            f"{stat}_season"
        )

    venue_matchup_stats = [
        "goals_for_venue_season",
        "goals_against_venue_season",
        "goal_diff_venue_season",
        "shots_for_venue_season",
        "shots_against_venue_season",
        "shot_diff_venue_season",
        "shooting_pct_venue_season",
        "save_pct_venue_season",
        "power_play_pct_venue_season",
        "win_venue_season"
    ]

    matchup_stats.extend(
        venue_matchup_stats
    )

    for stat in matchup_stats:
        features[
            f"{stat}_diff"
        ] = (
            features[
                f"home_{stat}"
            ]
            - features[
                f"away_{stat}"
            ]
        )

    features["rest_diff"] = (
        features["home_days_rest"]
        - features["away_days_rest"]
    )

    features["games_played_diff"] = (
        features["home_games_played"]
        - features["away_games_played"]
    )

    schedule_features = [
        "games_last_4_days",
        "games_last_6_days",
        "three_in_four",
        "four_in_six",
        "road_streak",
        "home_streak",
        "road_trip_game",
        "home_after_road_trip"
    ]

    for stat in schedule_features:
        features[
            f"{stat}_diff"
        ] = (
            features[
                f"home_{stat}"
            ]
            - features[
                f"away_{stat}"
            ]
        )

    features = features.drop(
        columns=[
            "home_feature_team_id",
            "away_feature_team_id"
        ]
    )

    return features


def main():
    games = pd.read_csv(
        INPUT_FILE
    )

    games["game_id"] = (
        games["game_id"]
        .astype(str)
    )

    games["date"] = pd.to_datetime(
        games["date"],
        utc=True
    )

    print(
        f"Loaded {len(games)} NHL games."
    )

    team_games = build_team_games(
        games
    )

    print(
        f"Built {len(team_games)} team-game rows."
    )

    team_games = add_team_features(
        team_games
    )

    features = build_game_features(
        games,
        team_games
    )

    features = features.sort_values(
        ["date", "game_id"]
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    features.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Feature rows: {len(features)}"
    )

    print(
        f"Feature columns: {len(features.columns)}"
    )

    print(
        f"Saved {OUTPUT_FILE}"
    )

    print(
        "\nSchedule feature sample:"
    )

    schedule_columns = [
        "home_back_to_back",
        "away_back_to_back",
        "home_three_in_four",
        "away_three_in_four",
        "home_four_in_six",
        "away_four_in_six",
        "home_road_trip_game",
        "away_road_trip_game",
        "home_home_after_road_trip",
        "away_home_after_road_trip"
    ]

    print(
        features[
            schedule_columns
        ]
        .tail(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()