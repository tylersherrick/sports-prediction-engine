import sys

import numpy as np
import pandas as pd


GOALIE_FILE = "data/processed/nhl_goalie_features.csv"

CAREER_WEIGHT = 0.30
SEASON_WEIGHT = 0.20
RECENT_WEIGHT = 0.20
LOCATION_WEIGHT = 0.15
REST_WEIGHT = 0.15

FULL_SPLIT_SAMPLE = 30


def weighted_save_pct(df):
    shots = df["shots_against"].sum()
    saves = df["saves"].sum()

    if shots <= 0:
        return np.nan

    return saves / shots


def split_stats(df):
    if df.empty:
        return None

    return {
        "games": len(df),
        "save_pct": weighted_save_pct(df),
        "avg_shots": df["shots_against"].mean(),
        "avg_saves": df["saves"].mean(),
        "avg_goals_against": df["goals_against"].mean()
    }


def print_split(name, stats):
    if stats is None:
        print(
            f"{name:<20} No games"
        )
        return

    print(
        f"{name:<20} "
        f"{stats['games']:>4} GP | "
        f"SV% {stats['save_pct']:.3f} | "
        f"Shots {stats['avg_shots']:.2f} | "
        f"Saves {stats['avg_saves']:.2f} | "
        f"GA {stats['avg_goals_against']:.2f}"
    )


def sample_factor(games):
    return min(
        games / FULL_SPLIT_SAMPLE,
        1.0
    )


def adjusted_weight(
    base_weight,
    stats
):
    if stats is None:
        return 0

    return (
        base_weight
        * sample_factor(
            stats["games"]
        )
    )


def rest_split(
    goalie,
    days_rest,
    back_to_back
):
    if back_to_back == 1:
        return (
            "Back-to-Back",
            split_stats(
                goalie[
                    goalie["back_to_back"] == 1
                ]
            )
        )

    if days_rest == 1:
        return (
            "1 Day Rest",
            split_stats(
                goalie[
                    goalie["days_rest"] == 1
                ]
            )
        )

    if days_rest == 2:
        return (
            "2 Days Rest",
            split_stats(
                goalie[
                    goalie["days_rest"] == 2
                ]
            )
        )

    return (
        "3+ Days Rest",
        split_stats(
            goalie[
                goalie["days_rest"] >= 3
            ]
        )
    )


def calculate_situational_save_pct(
    career,
    season,
    recent,
    location,
    rest
):
    components = [
        (
            "Career",
            career,
            CAREER_WEIGHT
        ),
        (
            "Current Season",
            season,
            SEASON_WEIGHT
        ),
        (
            "Last 10",
            recent,
            RECENT_WEIGHT
        ),
        (
            "Location",
            location,
            LOCATION_WEIGHT
        ),
        (
            "Rest",
            rest,
            REST_WEIGHT
        )
    ]

    numerator = 0
    denominator = 0
    details = []

    for (
        name,
        stats,
        base_weight
    ) in components:
        if stats is None:
            continue

        weight = adjusted_weight(
            base_weight,
            stats
        )

        if weight <= 0:
            continue

        numerator += (
            stats["save_pct"]
            * weight
        )

        denominator += weight

        details.append({
            "name": name,
            "games": stats["games"],
            "save_pct": stats["save_pct"],
            "weight": weight
        })

    if denominator == 0:
        return (
            np.nan,
            details
        )

    return (
        numerator / denominator,
        details
    )


def build_profile(goalie):
    goalie = goalie.sort_values(
        "date"
    ).copy()

    latest_season = goalie[
        "season"
    ].max()

    current = goalie[
        goalie["season"] == latest_season
    ]

    return {
        "career": split_stats(
            goalie
        ),
        "current": split_stats(
            current
        ),
        "last_5": split_stats(
            goalie.tail(5)
        ),
        "last_10": split_stats(
            goalie.tail(10)
        ),
        "home": split_stats(
            goalie[
                goalie["home"] == 1
            ]
        ),
        "away": split_stats(
            goalie[
                goalie["home"] == 0
            ]
        ),
        "back_to_back": split_stats(
            goalie[
                goalie["back_to_back"] == 1
            ]
        ),
        "rest_1": split_stats(
            goalie[
                goalie["days_rest"] == 1
            ]
        ),
        "rest_2": split_stats(
            goalie[
                goalie["days_rest"] == 2
            ]
        ),
        "rest_3_plus": split_stats(
            goalie[
                goalie["days_rest"] >= 3
            ]
        )
    }


def print_profile(
    player,
    team,
    profile
):
    print()
    print(
        "=" * 90
    )
    print(
        f"{player} | {team}"
    )
    print(
        "=" * 90
    )

    print()
    print("OVERALL")
    print("-" * 90)

    print_split(
        "Career",
        profile["career"]
    )

    print_split(
        "Current Season",
        profile["current"]
    )

    print_split(
        "Last 5",
        profile["last_5"]
    )

    print_split(
        "Last 10",
        profile["last_10"]
    )

    print()
    print("HOME / AWAY")
    print("-" * 90)

    print_split(
        "Home",
        profile["home"]
    )

    print_split(
        "Away",
        profile["away"]
    )

    print()
    print("REST")
    print("-" * 90)

    print_split(
        "Back-to-Back",
        profile["back_to_back"]
    )

    print_split(
        "1 Day Rest",
        profile["rest_1"]
    )

    print_split(
        "2 Days Rest",
        profile["rest_2"]
    )

    print_split(
        "3+ Days Rest",
        profile["rest_3_plus"]
    )


def print_situational_projection(
    goalie,
    profile
):
    latest = goalie.sort_values(
        "date"
    ).iloc[-1]

    is_home = int(
        latest["home"]
    )

    days_rest = latest[
        "days_rest"
    ]

    back_to_back = int(
        latest["back_to_back"]
    )

    if is_home == 1:
        location_name = "Home"
        location_stats = profile[
            "home"
        ]
    else:
        location_name = "Away"
        location_stats = profile[
            "away"
        ]

    rest_name, rest_stats = rest_split(
        goalie,
        days_rest,
        back_to_back
    )

    situational_pct, details = (
        calculate_situational_save_pct(
            profile["career"],
            profile["current"],
            profile["last_10"],
            location_stats,
            rest_stats
        )
    )

    opponent_shots = latest[
        "opponent_shots_for_last_5"
    ]

    if pd.isna(
        opponent_shots
    ):
        opponent_shots = latest[
            "opponent_shots_for_season"
        ]

    expected_saves = (
        situational_pct
        * opponent_shots
    )

    print()
    print("SITUATIONAL PROJECTION")
    print("-" * 90)

    print(
        f"Situation:          "
        f"{location_name} | "
        f"{rest_name}"
    )

    print(
        f"Opponent:           "
        f"{latest['opponent']}"
    )

    print(
        f"Expected Shots:     "
        f"{opponent_shots:.2f}"
    )

    print(
        f"Adjusted SV%:       "
        f"{situational_pct:.3f}"
    )

    print(
        f"Expected Saves:     "
        f"{expected_saves:.2f}"
    )

    print()
    print("SV% COMPONENTS")
    print("-" * 90)

    total_weight = sum(
        component["weight"]
        for component in details
    )

    for component in details:
        normalized_weight = (
            component["weight"]
            / total_weight
        )

        print(
            f"{component['name']:<20} "
            f"SV% {component['save_pct']:.3f} | "
            f"{component['games']:>3} GP | "
            f"Weight {normalized_weight:.1%}"
        )


def main():
    if len(sys.argv) < 2:
        print()
        print(
            'Usage: python -m '
            'src.evaluation.nhl_goalie_profile_report '
            '"Goalie Name"'
        )
        return

    search_name = " ".join(
        sys.argv[1:]
    ).strip()

    data = pd.read_csv(
        GOALIE_FILE,
        dtype={
            "game_id": str,
            "player_id": str,
            "team_id": str
        }
    )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "player_id",
            "player",
            "shots_against",
            "saves",
            "date"
        ]
    ).copy()

    data = data[
        data["shots_against"] > 0
    ].copy()

    matches = data[
        data["player"].str.contains(
            search_name,
            case=False,
            na=False
        )
    ].copy()

    if matches.empty:
        print()
        print(
            f'No goalie found matching '
            f'"{search_name}".'
        )
        return

    names = matches[
        "player"
    ].drop_duplicates()

    if len(names) > 1:
        print()
        print(
            "Multiple goalies found:"
        )

        for name in names:
            print(
                f"- {name}"
            )

        return

    player_name = names.iloc[
        0
    ]

    goalie = data[
        data["player"] == player_name
    ].copy()

    latest = goalie.sort_values(
        "date"
    ).iloc[-1]

    profile = build_profile(
        goalie
    )

    print_profile(
        player_name,
        latest["team"],
        profile
    )

    print_situational_projection(
        goalie,
        profile
    )


if __name__ == "__main__":
    main()