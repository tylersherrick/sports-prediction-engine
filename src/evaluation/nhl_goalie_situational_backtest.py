import numpy as np
import pandas as pd


GOALIE_FILE = "data/processed/nhl_goalie_features.csv"
TEAM_FILE = "data/processed/nhl_features.csv"

MIN_PRIOR_GAMES = 5
TARGET_TOI = 3000
HISTORY_TOI = 3000
FULL_SPLIT_SAMPLE = 30

GOALIE_WINDOW = 15
SAVE_PCT_WINDOW = 20

CAREER_WEIGHT = 0.20
SEASON_WEIGHT = 0.35
RECENT_WEIGHT = 0.45

GOALIE_WEIGHT = 0.50
OPPONENT_WEIGHT = 0.50

OPPONENT_LAST_5_WEIGHT = 0.10
OPPONENT_SEASON_WEIGHT = 0.90

BLOCK_WEIGHT = 0.17

VENUE_WEIGHT = 0.60
RECENT_WORKLOAD_WEIGHT = 0.40

VENUE_WINDOWS = [
    5,
    10,
    15,
    20,
    30
]


def weighted_save_pct(df):
    shots = df["shots_against"].sum()
    saves = df["saves"].sum()

    if shots <= 0:
        return np.nan

    return saves / shots


def sample_factor(games):
    return min(
        games / FULL_SPLIT_SAMPLE,
        1.0
    )


def add_component(
    components,
    df,
    base_weight
):
    if df.empty or base_weight <= 0:
        return

    save_pct = weighted_save_pct(df)

    if pd.isna(save_pct):
        return

    weight = (
        base_weight
        * sample_factor(len(df))
    )

    if weight <= 0:
        return

    components.append(
        (
            save_pct,
            weight
        )
    )


def performance_save_pct(
    history,
    current
):
    season_history = history[
        history["season"]
        == current["season"]
    ]

    recent_history = history.tail(
        SAVE_PCT_WINDOW
    )

    components = []

    add_component(
        components,
        history,
        CAREER_WEIGHT
    )

    add_component(
        components,
        season_history,
        SEASON_WEIGHT
    )

    add_component(
        components,
        recent_history,
        RECENT_WEIGHT
    )

    if not components:
        return np.nan

    total_weight = sum(
        weight
        for _, weight in components
    )

    return sum(
        save_pct * weight
        for save_pct, weight in components
    ) / total_weight


def goalie_recent_shots(history):
    recent = history.tail(
        GOALIE_WINDOW
    )

    if recent.empty:
        return np.nan

    return recent[
        "shots_against"
    ].mean()


def goalie_venue_shots(
    history,
    current,
    window
):
    venue_history = history[
        history["home"]
        == current["home"]
    ].tail(
        window
    )

    if venue_history.empty:
        return np.nan

    return venue_history[
        "shots_against"
    ].mean()


def opponent_shot_projection(current):
    last_5 = current[
        "opponent_shots_for_last_5"
    ]

    season = current[
        "opponent_shots_for_season"
    ]

    if (
        pd.isna(last_5)
        or pd.isna(season)
    ):
        return np.nan

    return (
        OPPONENT_LAST_5_WEIGHT
        * last_5
        + OPPONENT_SEASON_WEIGHT
        * season
    )


def blocked_shot_projection(current):
    attempts = current[
        "opponent_shot_attempts_for_last_10"
    ]

    through_pct = current[
        "opponent_shot_through_pct_last_10"
    ]

    if (
        pd.isna(attempts)
        or pd.isna(through_pct)
    ):
        return np.nan

    return attempts * through_pct


def run_backtest(data):
    rows = []

    for player_id, goalie in data.groupby(
        "player_id"
    ):
        goalie = goalie.sort_values(
            [
                "date",
                "game_id"
            ]
        ).reset_index(
            drop=True
        )

        for index in range(
            len(goalie)
        ):
            current = goalie.iloc[index]

            if (
                current["time_on_ice"]
                < TARGET_TOI
            ):
                continue

            prior = goalie.iloc[
                :index
            ].copy()

            history = prior[
                prior["time_on_ice"]
                >= HISTORY_TOI
            ].copy()

            if len(history) < MIN_PRIOR_GAMES:
                continue

            save_pct = performance_save_pct(
                history,
                current
            )

            goalie_shots = goalie_recent_shots(
                history
            )

            opponent_shots = (
                opponent_shot_projection(
                    current
                )
            )

            blocked_projection = (
                blocked_shot_projection(
                    current
                )
            )

            if (
                pd.isna(save_pct)
                or pd.isna(goalie_shots)
                or pd.isna(opponent_shots)
                or pd.isna(blocked_projection)
            ):
                continue

            row = {
                "game_id": current["game_id"],
                "season": current["season"],
                "player_id": current["player_id"],
                "actual_shots": current[
                    "shots_against"
                ],
                "actual_saves": current[
                    "saves"
                ]
            }

            valid = True

            for window in VENUE_WINDOWS:
                venue_shots = goalie_venue_shots(
                    history,
                    current,
                    window
                )

                if pd.isna(venue_shots):
                    valid = False
                    break

                adjusted_goalie_shots = (
                    RECENT_WORKLOAD_WEIGHT
                    * goalie_shots
                    + VENUE_WEIGHT
                    * venue_shots
                )

                base_projection = (
                    GOALIE_WEIGHT
                    * adjusted_goalie_shots
                    + OPPONENT_WEIGHT
                    * opponent_shots
                )

                expected_shots = (
                    (1.0 - BLOCK_WEIGHT)
                    * base_projection
                    + BLOCK_WEIGHT
                    * blocked_projection
                )

                row[
                    f"shots_{window}"
                ] = expected_shots

                row[
                    f"saves_{window}"
                ] = (
                    expected_shots
                    * save_pct
                )

            if valid:
                rows.append(row)

    return pd.DataFrame(rows)


def mae(
    actual,
    predicted
):
    return np.mean(
        np.abs(
            actual - predicted
        )
    )


def evaluate(
    results,
    title
):
    print()
    print(title)
    print("=" * 100)

    rows = []

    for window in VENUE_WINDOWS:
        saves_mae = mae(
            results["actual_saves"],
            results[
                f"saves_{window}"
            ]
        )

        shots_mae = mae(
            results["actual_shots"],
            results[
                f"shots_{window}"
            ]
        )

        rows.append({
            "window": window,
            "games": len(results),
            "saves_mae": saves_mae,
            "shots_mae": shots_mae
        })

    for row in rows:
        print(
            f"Venue Last {row['window']:>2} | "
            f"Games {row['games']:>6,} | "
            f"Saves MAE {row['saves_mae']:.4f} | "
            f"Shots MAE {row['shots_mae']:.4f}"
        )

    best = min(
        rows,
        key=lambda row:
        row["saves_mae"]
    )

    print()
    print("BEST")
    print("-" * 100)

    print(
        f"Venue window: "
        f"Last {best['window']}"
    )

    print(
        f"Games:        "
        f"{best['games']:,}"
    )

    print(
        f"Saves MAE:    "
        f"{best['saves_mae']:.4f}"
    )

    print(
        f"Shots MAE:    "
        f"{best['shots_mae']:.4f}"
    )


def main():
    goalies = pd.read_csv(
        GOALIE_FILE,
        dtype={
            "game_id": str,
            "player_id": str,
            "team_id": str,
            "opponent_team_id": str
        }
    )

    teams = pd.read_csv(
        TEAM_FILE,
        dtype={
            "game_id": str,
            "home_team_id": str,
            "away_team_id": str
        }
    )

    goalies["date"] = pd.to_datetime(
        goalies["date"],
        errors="coerce"
    )

    teams = teams[
        [
            "game_id",
            "home_team_id",
            "away_team_id",
            "home_shot_attempts_for_last_10",
            "away_shot_attempts_for_last_10",
            "home_shot_through_pct_last_10",
            "away_shot_through_pct_last_10"
        ]
    ].copy()

    home = teams[
        [
            "game_id",
            "home_team_id",
            "home_shot_attempts_for_last_10",
            "home_shot_through_pct_last_10"
        ]
    ].rename(
        columns={
            "home_team_id":
            "opponent_team_id",
            "home_shot_attempts_for_last_10":
            "opponent_shot_attempts_for_last_10",
            "home_shot_through_pct_last_10":
            "opponent_shot_through_pct_last_10"
        }
    )

    away = teams[
        [
            "game_id",
            "away_team_id",
            "away_shot_attempts_for_last_10",
            "away_shot_through_pct_last_10"
        ]
    ].rename(
        columns={
            "away_team_id":
            "opponent_team_id",
            "away_shot_attempts_for_last_10":
            "opponent_shot_attempts_for_last_10",
            "away_shot_through_pct_last_10":
            "opponent_shot_through_pct_last_10"
        }
    )

    opponent_features = pd.concat(
        [
            home,
            away
        ],
        ignore_index=True
    )

    goalies = goalies.merge(
        opponent_features,
        on=[
            "game_id",
            "opponent_team_id"
        ],
        how="left"
    )

    numeric_columns = [
        "shots_against",
        "saves",
        "time_on_ice",
        "home",
        "opponent_shots_for_last_5",
        "opponent_shots_for_season",
        "opponent_shot_attempts_for_last_10",
        "opponent_shot_through_pct_last_10"
    ]

    for column in numeric_columns:
        goalies[column] = pd.to_numeric(
            goalies[column],
            errors="coerce"
        )

    goalies = goalies.dropna(
        subset=[
            "player_id",
            "date",
            "shots_against",
            "saves",
            "time_on_ice",
            "home"
        ]
    ).copy()

    goalies = goalies[
        goalies["shots_against"] > 0
    ].copy()

    print()
    print(
        "NHL GOALIE VENUE WINDOW TEST"
    )

    print(
        "Historical appearances: 50+ minutes"
    )

    print(
        "Overall goalie workload: Last 15"
    )

    print(
        "Workload blend: 60% same venue / 40% overall"
    )

    results = run_backtest(
        goalies
    )

    evaluate(
        results,
        "ALL HISTORICAL SEASONS"
    )

    latest_season = goalies[
        "season"
    ].max()

    current = results[
        results["season"]
        == latest_season
    ].copy()

    evaluate(
        current,
        f"CURRENT SEASON ({latest_season})"
    )


if __name__ == "__main__":
    main()