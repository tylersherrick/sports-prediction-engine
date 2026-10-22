import re

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


GOALIE_FILE = (
    "data/processed/"
    "nhl_moneypuck_goalie_features.csv"
)

TEAM_FILE = (
    "data/processed/"
    "nhl_moneypuck_team_features.csv"
)

KALSHI_FILE = (
    "data/processed/"
    "nhl_kalshi_saves_history.csv"
)


TEST_SEASON = 2026

CURRENT_SEASON_WEIGHT = 2.0
CURRENT_SEASON_DELAY_DAYS = 7


def normalize_name(name):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(name).lower()
    )


def find_column(
    df,
    candidates
):
    for column in candidates:
        if column in df.columns:
            return column

    return None


def safe_features(
    df,
    metric
):
    allowed_suffixes = [
        "_season",
        "_last_5",
        "_last_10",
        "_last_15",
        "_home_last_5",
        "_home_last_10",
        "_home_last_15",
        "_away_last_5",
        "_away_last_10",
        "_away_last_15"
    ]

    features = []

    metric_lower = metric.lower()

    for column in df.columns:
        lower = column.lower()

        if metric_lower not in lower:
            continue

        if any(
            lower.endswith(
                suffix.lower()
            )
            for suffix in allowed_suffixes
        ):
            features.append(
                column
            )

    return features


def load_goalies():
    goalies = pd.read_csv(
        GOALIE_FILE,
        low_memory=False
    ).copy()

    goalies["game_date"] = pd.to_datetime(
        goalies["game_date"],
        errors="coerce"
    ).dt.normalize()

    goalies["season"] = pd.to_numeric(
        goalies["season"],
        errors="coerce"
    )

    goalies["player_key"] = (
        goalies["name"]
        .apply(normalize_name)
    )

    return goalies


def load_teams():
    teams = pd.read_csv(
        TEAM_FILE,
        low_memory=False
    ).copy()

    teams["game_date"] = pd.to_datetime(
        teams["game_date"],
        errors="coerce"
    ).dt.normalize()

    return teams


def build_dataset():
    goalies = load_goalies()
    teams = load_teams()

    target_column = find_column(
        goalies,
        [
            "saves",
            "goalie_saves"
        ]
    )

    opponent_column = find_column(
        goalies,
        [
            "opposingTeam",
            "opponent",
            "opponent_team"
        ]
    )

    team_column = find_column(
        teams,
        [
            "team",
            "playerTeam",
            "teamCode",
            "team_key"
        ]
    )

    if target_column is None:
        raise RuntimeError(
            "Could not find saves column."
        )

    if opponent_column is None:
        raise RuntimeError(
            "Could not find opponent column."
        )

    if team_column is None:
        raise RuntimeError(
            "Could not find team column."
        )

    baseline_goalie = [
        column
        for column in [
            "goalie_ongoal_last_5",
            "goalie_ongoal_last_10",
            "goalie_ongoal_last_15",

            "goalie_saves_last_5",
            "goalie_saves_last_10",
            "goalie_saves_last_15",

            "goalie_save_pct_last_5",
            "goalie_save_pct_last_10",
            "goalie_save_pct_last_15",

            "goalie_gsax_last_5",
            "goalie_gsax_last_10",
            "goalie_gsax_last_15"
        ]
        if column in goalies.columns
    ]

    team_shots = [
        column
        for column in [
            "team_shotsOnGoalFor_last_5",
            "team_shotsOnGoalFor_last_10",
            "team_shotsOnGoalFor_last_15"
        ]
        if column in teams.columns
    ]

    low_shots = safe_features(
        goalies,
        "goalie_lowDangerShots"
    )

    low_xgoals = safe_features(
        goalies,
        "goalie_lowDangerxGoals"
    )

    medium_shots = safe_features(
        goalies,
        "goalie_mediumDangerShots"
    )

    medium_xgoals = safe_features(
        goalies,
        "goalie_mediumDangerxGoals"
    )

    high_shots = safe_features(
        goalies,
        "goalie_highDangerShots"
    )

    high_xgoals = safe_features(
        goalies,
        "goalie_highDangerxGoals"
    )

    rebounds = safe_features(
        goalies,
        "goalie_rebounds_above_expected"
    )

    low_combined = list(
        dict.fromkeys(
            low_shots
            + low_xgoals
        )
    )

    medium_combined = list(
        dict.fromkeys(
            medium_shots
            + medium_xgoals
        )
    )

    high_combined = list(
        dict.fromkeys(
            high_shots
            + high_xgoals
        )
    )

    all_extra = list(
        dict.fromkeys(
            low_combined
            + medium_combined
            + high_combined
            + rebounds
        )
    )

    goalie_columns = [
        "game_date",
        "season",
        "name",
        "player_key",
        opponent_column,
        target_column
    ]

    goalie_columns += (
        baseline_goalie
        + all_extra
    )

    goalie_columns = list(
        dict.fromkeys(
            goalie_columns
        )
    )

    goalie_data = goalies[
        goalie_columns
    ].copy()

    goalie_data = goalie_data.rename(
        columns={
            opponent_column:
                "opponent",

            target_column:
                "actual_saves"
        }
    )

    goalie_data["opponent"] = (
        goalie_data["opponent"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    team_columns = [
        "game_date",
        team_column
    ]

    team_columns += team_shots

    team_data = teams[
        team_columns
    ].copy()

    team_data = team_data.rename(
        columns={
            team_column:
                "opponent"
        }
    )

    team_data["opponent"] = (
        team_data["opponent"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    data = goalie_data.merge(
        team_data,
        on=[
            "game_date",
            "opponent"
        ],
        how="left"
    )

    baseline = list(
        dict.fromkeys(
            baseline_goalie
            + team_shots
        )
    )

    feature_sets = {
        "BASELINE":
            baseline,

        "LOW DANGER SHOTS":
            list(
                dict.fromkeys(
                    baseline
                    + low_shots
                )
            ),

        "MEDIUM DANGER":
            list(
                dict.fromkeys(
                    baseline
                    + medium_combined
                )
            ),

        "HIGH DANGER":
            list(
                dict.fromkeys(
                    baseline
                    + high_combined
                )
            ),

        "LOW + MEDIUM":
            list(
                dict.fromkeys(
                    baseline
                    + low_shots
                    + medium_combined
                )
            ),

        "LOW + HIGH":
            list(
                dict.fromkeys(
                    baseline
                    + low_shots
                    + high_combined
                )
            )
    }

    all_features = []

    for features in feature_sets.values():
        all_features.extend(
            features
        )

    all_features = list(
        dict.fromkeys(
            all_features
        )
    )

    for column in (
        all_features
        + [
            "actual_saves"
        ]
    ):
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna(
        subset=[
            "game_date",
            "season",
            "name",
            "actual_saves"
        ]
    ).copy()

    data = data.sort_values(
        [
            "game_date",
            "name"
        ]
    ).reset_index(
        drop=True
    )

    return (
        data,
        feature_sets
    )


def prepare_kalshi():
    kalshi = pd.read_csv(
        KALSHI_FILE
    )

    kalshi["market_date"] = pd.to_datetime(
        kalshi["game_date"],
        errors="coerce"
    ).dt.normalize()

    kalshi["player_key"] = (
        kalshi["player"]
        .apply(normalize_name)
    )

    kalshi["threshold"] = pd.to_numeric(
        kalshi["threshold"],
        errors="coerce"
    )

    kalshi["comparison_line"] = (
        kalshi["threshold"]
        - 0.5
    )

    kalshi["result"] = (
        kalshi["result"]
        .astype(str)
        .str.lower()
    )

    kalshi = kalshi.dropna(
        subset=[
            "market_date",
            "player_key",
            "threshold"
        ]
    ).copy()

    kalshi = kalshi[
        kalshi["result"].isin(
            [
                "yes",
                "no"
            ]
        )
    ].copy()

    return kalshi


def select_markets(
    kalshi
):
    markets = kalshi[
        [
            "ticker",
            "player",
            "player_key",
            "market_date",
            "threshold",
            "comparison_line",
            "result"
        ]
    ].copy()

    markets["threshold_rank"] = (
        markets.groupby(
            [
                "player_key",
                "market_date"
            ]
        )["threshold"]
        .rank(
            method="first",
            pct=True
        )
    )

    markets[
        "distance_from_middle"
    ] = (
        markets["threshold_rank"]
        - 0.5
    ).abs()

    markets = (
        markets.sort_values(
            [
                "player_key",
                "market_date",
                "distance_from_middle",
                "threshold"
            ]
        )
        .drop_duplicates(
            subset=[
                "player_key",
                "market_date"
            ],
            keep="first"
        )
        .reset_index(
            drop=True
        )
    )

    return markets


def make_model():
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                Ridge(
                    alpha=10.0
                )
            )
        ]
    )


def run_test_one(
    data,
    features
):
    train = data[
        data["season"]
        < TEST_SEASON
    ].copy()

    test = data[
        data["season"]
        == TEST_SEASON
    ].copy()

    model = make_model()

    model.fit(
        train[
            features
        ],
        train[
            "actual_saves"
        ]
    )

    test["projection"] = (
        model.predict(
            test[
                features
            ]
        )
    )

    return test


def run_test_two(
    data,
    features
):
    historical = data[
        data["season"]
        < TEST_SEASON
    ].copy()

    current = data[
        data["season"]
        == TEST_SEASON
    ].copy()

    predictions = []

    for _, game in current.iterrows():
        cutoff = (
            game["game_date"]
            - pd.Timedelta(
                days=CURRENT_SEASON_DELAY_DAYS
            )
        )

        current_training = current[
            current["game_date"]
            <= cutoff
        ].copy()

        train = pd.concat(
            [
                historical,
                current_training
            ],
            ignore_index=True
        )

        weights = np.where(
            train["season"]
            == TEST_SEASON,
            CURRENT_SEASON_WEIGHT,
            1.0
        )

        model = make_model()

        model.fit(
            train[
                features
            ],
            train[
                "actual_saves"
            ],
            model__sample_weight=weights
        )

        prediction = model.predict(
            pd.DataFrame(
                [
                    game[
                        features
                    ].to_dict()
                ]
            )
        )[0]

        row = game.copy()

        row["projection"] = (
            prediction
        )

        predictions.append(
            row
        )

    return pd.DataFrame(
        predictions
    )


def grade(
    predictions,
    markets
):
    predictions = (
        predictions.copy()
    )

    predictions["game_date"] = (
        pd.to_datetime(
            predictions["game_date"],
            errors="coerce"
        )
        .dt.normalize()
    )

    graded = predictions.merge(
        markets,
        left_on=[
            "player_key",
            "game_date"
        ],
        right_on=[
            "player_key",
            "market_date"
        ],
        how="inner"
    )

    graded["edge"] = (
        graded["projection"]
        - graded["comparison_line"]
    )

    graded["abs_edge"] = (
        graded["edge"].abs()
    )

    graded["pick"] = np.where(
        graded["edge"] >= 0,
        "yes",
        "no"
    )

    graded["correct"] = (
        graded["pick"]
        == graded["result"]
    )

    return graded


def record(
    graded
):
    total = len(
        graded
    )

    wins = int(
        graded["correct"].sum()
    )

    losses = (
        total
        - wins
    )

    rate = (
        wins / total
        if total
        else 0
    )

    return (
        wins,
        losses,
        rate,
        total
    )


def no_bucket_record(
    graded,
    minimum,
    maximum
):
    subset = graded[
        (
            graded["pick"]
            == "no"
        )
        & (
            graded["abs_edge"]
            >= minimum
        )
        & (
            graded["abs_edge"]
            < maximum
        )
    ].copy()

    return record(
        subset
    )


def print_record(
    label,
    result
):
    print(
        f"    {label}: "
        f"{result[0]}-{result[1]} | "
        f"{result[2]:.1%} | "
        f"{result[3]} bets"
    )


def print_test_breakdown(
    label,
    graded
):
    print()
    print(
        f"  {label} - NO BETS"
    )

    print_record(
        "3.0-3.5",
        no_bucket_record(
            graded,
            3.0,
            3.5
        )
    )

    print_record(
        "3.5-4.0",
        no_bucket_record(
            graded,
            3.5,
            4.0
        )
    )

    print_record(
        "4.0-4.5",
        no_bucket_record(
            graded,
            4.0,
            4.5
        )
    )

    print_record(
        "4.5-5.0",
        no_bucket_record(
            graded,
            4.5,
            5.0
        )
    )

    print_record(
        "3.0-5.0 TOTAL",
        no_bucket_record(
            graded,
            3.0,
            5.0
        )
    )


def print_result(
    name,
    features,
    test_one,
    test_two
):
    print()
    print("=" * 70)

    print(
        f"{name} | "
        f"{len(features)} features"
    )

    print_test_breakdown(
        "TEST 1",
        test_one
    )

    print_test_breakdown(
        "TEST 2",
        test_two
    )


def main():
    print()
    print(
        "NHL GOALIE SAVES - NO EDGE 3 TO 5 DEEP DIVE"
    )
    print("=" * 70)

    data, feature_sets = (
        build_dataset()
    )

    markets = select_markets(
        prepare_kalshi()
    )

    for (
        name,
        features
    ) in feature_sets.items():

        test_one = grade(
            run_test_one(
                data,
                features
            ),
            markets
        )

        test_two = grade(
            run_test_two(
                data,
                features
            ),
            markets
        )

        print_result(
            name,
            features,
            test_one,
            test_two
        )


if __name__ == "__main__":
    main()