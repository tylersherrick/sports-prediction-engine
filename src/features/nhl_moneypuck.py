from pathlib import Path

import numpy as np
import pandas as pd


TEAM_FILE = Path(
    "data/processed/nhl_moneypuck_team_games.csv"
)

GOALIE_FILE = Path(
    "data/processed/nhl_moneypuck_goalie_games.csv"
)

TEAM_OUTPUT_FILE = Path(
    "data/processed/nhl_moneypuck_team_features.csv"
)

GOALIE_OUTPUT_FILE = Path(
    "data/processed/nhl_moneypuck_goalie_features.csv"
)


WINDOWS = [
    5,
    10,
    15
]


def clean_game_date(series):
    text = (
        series
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True
        )
    )

    return pd.to_datetime(
        text,
        format="%Y%m%d",
        errors="coerce"
    )


def filter_all_situations(frame):
    if "situation" not in frame.columns:
        return frame.copy()

    situations = (
        frame["situation"]
        .dropna()
        .astype(str)
        .str.lower()
        .unique()
    )

    if "all" in situations:
        return frame[
            frame["situation"]
            .astype(str)
            .str.lower()
            .eq("all")
        ].copy()

    return frame.copy()


def numeric(frame, column):
    if column not in frame.columns:
        return pd.Series(
            np.nan,
            index=frame.index
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce"
    )


def add_rest_features(
    frame,
    group_column
):
    frame = frame.sort_values(
        [
            group_column,
            "game_date",
            "gameId"
        ]
    ).copy()

    frame["previous_game_date"] = (
        frame
        .groupby(group_column)[
            "game_date"
        ]
        .shift(1)
    )

    frame["days_since_previous_game"] = (
        frame["game_date"]
        - frame["previous_game_date"]
    ).dt.days

    frame["days_rest"] = (
        frame[
            "days_since_previous_game"
        ]
        - 1
    )

    frame["days_rest"] = (
        frame["days_rest"]
        .clip(lower=0)
    )

    frame["back_to_back"] = (
        frame[
            "days_since_previous_game"
        ]
        .eq(1)
        .astype(int)
    )

    return frame


def add_rolling_features(
    frame,
    group_column,
    metrics,
    prefix
):
    frame = frame.sort_values(
        [
            group_column,
            "game_date",
            "gameId"
        ]
    ).copy()

    for metric in metrics:
        if metric not in frame.columns:
            continue

        frame[metric] = pd.to_numeric(
            frame[metric],
            errors="coerce"
        )

        grouped = frame.groupby(
            group_column
        )[metric]

        frame[
            f"{prefix}_{metric}_season"
        ] = grouped.transform(
            lambda values: (
                values
                .shift(1)
                .expanding()
                .mean()
            )
        )

        for window in WINDOWS:
            frame[
                f"{prefix}_{metric}_last_{window}"
            ] = grouped.transform(
                lambda values,
                window=window: (
                    values
                    .shift(1)
                    .rolling(
                        window,
                        min_periods=1
                    )
                    .mean()
                )
            )

    return frame


def add_split_features(
    frame,
    group_column,
    metrics,
    prefix
):
    if "home_or_away" not in frame.columns:
        return frame

    frame = frame.sort_values(
        [
            group_column,
            "game_date",
            "gameId"
        ]
    ).copy()

    for metric in metrics:
        if metric not in frame.columns:
            continue

        frame[metric] = pd.to_numeric(
            frame[metric],
            errors="coerce"
        )

        for venue in [
            "HOME",
            "AWAY"
        ]:
            venue_key = venue.lower()

            values = frame[metric].where(
                frame[
                    "home_or_away"
                ]
                .astype(str)
                .str.upper()
                .eq(venue)
            )

            temporary = pd.DataFrame(
                {
                    group_column:
                        frame[group_column],
                    "value":
                        values
                },
                index=frame.index
            )

            grouped = temporary.groupby(
                group_column
            )["value"]

            for window in WINDOWS:
                frame[
                    (
                        f"{prefix}_{metric}_"
                        f"{venue_key}_last_{window}"
                    )
                ] = grouped.transform(
                    lambda values,
                    window=window: (
                        values
                        .shift(1)
                        .rolling(
                            window,
                            min_periods=1
                        )
                        .mean()
                    )
                )

    return frame


def build_goalie_features():
    print()
    print("BUILDING MONEYPUCK GOALIE FEATURES")
    print("=" * 60)

    goalies = pd.read_csv(
        GOALIE_FILE,
        dtype={
            "playerId": str,
            "gameId": str
        }
    )

    goalies = filter_all_situations(
        goalies
    )

    goalies["game_date"] = clean_game_date(
        goalies["gameDate"]
    )

    goalies["icetime"] = numeric(
        goalies,
        "icetime"
    )

    goalies["xGoals"] = numeric(
        goalies,
        "xGoals"
    )

    goalies["goals"] = numeric(
        goalies,
        "goals"
    )

    goalies["ongoal"] = numeric(
        goalies,
        "ongoal"
    )

    goalies["xOnGoal"] = numeric(
        goalies,
        "xOnGoal"
    )

    goalies["rebounds"] = numeric(
        goalies,
        "rebounds"
    )

    goalies["xRebounds"] = numeric(
        goalies,
        "xRebounds"
    )

    goalies["saves"] = (
        goalies["ongoal"]
        - goalies["goals"]
    )

    goalies["gsax"] = (
        goalies["xGoals"]
        - goalies["goals"]
    )

    goalies["gsax_per_60"] = np.where(
        goalies["icetime"] > 0,
        (
            goalies["gsax"]
            * 3600
            / goalies["icetime"]
        ),
        np.nan
    )

    goalies["save_pct"] = np.where(
        goalies["ongoal"] > 0,
        (
            goalies["saves"]
            / goalies["ongoal"]
        ),
        np.nan
    )

    goalies["expected_save_pct"] = np.where(
        goalies["ongoal"] > 0,
        (
            (
                goalies["ongoal"]
                - goalies["xGoals"]
            )
            / goalies["ongoal"]
        ),
        np.nan
    )

    goalies["save_pct_above_expected"] = (
        goalies["save_pct"]
        - goalies["expected_save_pct"]
    )

    goalies["rebounds_above_expected"] = (
        goalies["rebounds"]
        - goalies["xRebounds"]
    )

    for danger in [
        "lowDanger",
        "mediumDanger",
        "highDanger"
    ]:
        shots_column = (
            f"{danger}Shots"
        )

        goals_column = (
            f"{danger}Goals"
        )

        xgoals_column = (
            f"{danger}xGoals"
        )

        if (
            shots_column
            in goalies.columns
        ):
            goalies[
                shots_column
            ] = numeric(
                goalies,
                shots_column
            )

        if (
            goals_column
            in goalies.columns
        ):
            goalies[
                goals_column
            ] = numeric(
                goalies,
                goals_column
            )

        if (
            xgoals_column
            in goalies.columns
        ):
            goalies[
                xgoals_column
            ] = numeric(
                goalies,
                xgoals_column
            )

        if (
            xgoals_column
            in goalies.columns
            and goals_column
            in goalies.columns
        ):
            goalies[
                f"{danger}_gsax"
            ] = (
                goalies[
                    xgoals_column
                ]
                - goalies[
                    goals_column
                ]
            )

    goalie_metrics = [
        "saves",
        "ongoal",
        "xOnGoal",
        "goals",
        "xGoals",
        "gsax",
        "gsax_per_60",
        "save_pct",
        "expected_save_pct",
        "save_pct_above_expected",
        "rebounds",
        "xRebounds",
        "rebounds_above_expected",
        "lowDangerShots",
        "mediumDangerShots",
        "highDangerShots",
        "lowDangerxGoals",
        "mediumDangerxGoals",
        "highDangerxGoals",
        "lowDanger_gsax",
        "mediumDanger_gsax",
        "highDanger_gsax"
    ]

    goalie_metrics = [
        metric
        for metric in goalie_metrics
        if metric in goalies.columns
    ]

    goalies = add_rest_features(
        goalies,
        "playerId"
    )

    goalies = add_rolling_features(
        goalies,
        "playerId",
        goalie_metrics,
        "goalie"
    )

    goalies = add_split_features(
        goalies,
        "playerId",
        goalie_metrics,
        "goalie"
    )

    goalies = goalies.sort_values(
        [
            "game_date",
            "gameId",
            "name"
        ]
    ).reset_index(
        drop=True
    )

    goalies.to_csv(
        GOALIE_OUTPUT_FILE,
        index=False
    )

    print(
        f"Rows: {len(goalies):,}"
    )

    print(
        f"Goalies: "
        f"{goalies['playerId'].nunique():,}"
    )

    print(
        f"Features: "
        f"{len(goalies.columns):,}"
    )

    print(
        f"Saved: "
        f"{GOALIE_OUTPUT_FILE}"
    )


def find_team_column(frame):
    for column in [
        "team",
        "playerTeam",
        "teamCode"
    ]:
        if column in frame.columns:
            return column

    raise RuntimeError(
        "Could not identify MoneyPuck team column."
    )


def build_team_features():
    print()
    print("BUILDING MONEYPUCK TEAM FEATURES")
    print("=" * 60)

    teams = pd.read_csv(
        TEAM_FILE,
        dtype={
            "gameId": str
        }
    )

    teams = filter_all_situations(
        teams
    )

    team_column = find_team_column(
        teams
    )

    teams["team_key"] = (
        teams[team_column]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    if "gameDate" not in teams.columns:
        raise RuntimeError(
            "MoneyPuck team file does not contain gameDate."
        )

    teams["game_date"] = clean_game_date(
        teams["gameDate"]
    )

    possible_metrics = [
        "goalsFor",
        "goalsAgainst",
        "shotsOnGoalFor",
        "shotsOnGoalAgainst",
        "shotAttemptsFor",
        "shotAttemptsAgainst",
        "unblockedShotAttemptsFor",
        "unblockedShotAttemptsAgainst",
        "xGoalsFor",
        "xGoalsAgainst",
        "flurryAdjustedxGoalsFor",
        "flurryAdjustedxGoalsAgainst",
        "lowDangerShotsFor",
        "lowDangerShotsAgainst",
        "mediumDangerShotsFor",
        "mediumDangerShotsAgainst",
        "highDangerShotsFor",
        "highDangerShotsAgainst",
        "lowDangerxGoalsFor",
        "lowDangerxGoalsAgainst",
        "mediumDangerxGoalsFor",
        "mediumDangerxGoalsAgainst",
        "highDangerxGoalsFor",
        "highDangerxGoalsAgainst",
        "reboundsFor",
        "reboundsAgainst",
        "xReboundsFor",
        "xReboundsAgainst",
        "penaltiesFor",
        "penaltiesAgainst"
    ]

    team_metrics = [
        metric
        for metric in possible_metrics
        if metric in teams.columns
    ]

    if not team_metrics:
        excluded = {
            "season",
            "gameId",
            "gameDate",
            "situation",
            "home_or_away",
            team_column,
            "team_key"
        }

        for column in teams.columns:
            if column in excluded:
                continue

            converted = pd.to_numeric(
                teams[column],
                errors="coerce"
            )

            if converted.notna().any():
                teams[column] = converted
                team_metrics.append(
                    column
                )

    print(
        f"Team metrics found: "
        f"{len(team_metrics):,}"
    )

    teams = add_rest_features(
        teams,
        "team_key"
    )

    teams = add_rolling_features(
        teams,
        "team_key",
        team_metrics,
        "team"
    )

    teams = add_split_features(
        teams,
        "team_key",
        team_metrics,
        "team"
    )

    teams = teams.sort_values(
        [
            "game_date",
            "gameId",
            "team_key"
        ]
    ).reset_index(
        drop=True
    )

    teams.to_csv(
        TEAM_OUTPUT_FILE,
        index=False
    )

    print(
        f"Rows: {len(teams):,}"
    )

    print(
        f"Teams: "
        f"{teams['team_key'].nunique():,}"
    )

    print(
        f"Features: "
        f"{len(teams.columns):,}"
    )

    print(
        f"Saved: "
        f"{TEAM_OUTPUT_FILE}"
    )


def main():
    print()
    print("MONEYPUCK NHL FEATURE BUILDER")
    print("=" * 60)

    build_goalie_features()
    build_team_features()

    print()
    print("=" * 60)
    print("MONEYPUCK FEATURES COMPLETE")


if __name__ == "__main__":
    main()