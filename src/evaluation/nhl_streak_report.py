import pandas as pd


SKATER_FILE = "data/processed/nhl_skater_features.csv"

PROPS = {
    "Points": "points",
    "Goals": "goals",
    "Assists": "assists",
    "Shots on Goal": "shots_on_goal"
}

MAX_STREAK = 5
MIN_SAMPLE = 20


def analyze_binary_prop(
    data,
    prop_name,
    target
):
    rows = []

    for player_id, player in data.groupby(
        "player_id"
    ):
        player = player.sort_values(
            ["season", "game_id"]
        ).copy()

        values = player[
            target
        ].fillna(0).to_numpy()

        seasons = player[
            "season"
        ].to_numpy()

        for index in range(
            1,
            len(player)
        ):
            current_season = seasons[
                index
            ]

            streak = 0

            for previous in range(
                index - 1,
                -1,
                -1
            ):
                if (
                    seasons[previous]
                    != current_season
                ):
                    break

                if values[previous] == 0:
                    streak += 1
                else:
                    break

            if streak < 1:
                continue

            streak = min(
                streak,
                MAX_STREAK
            )

            hit = (
                values[index] > 0
            )

            rows.append({
                "player_id": player_id,
                "prop": prop_name,
                "streak": streak,
                "hit": hit
            })

    return pd.DataFrame(
        rows
    )


def analyze_shots(data):
    rows = []

    for player_id, player in data.groupby(
        "player_id"
    ):
        player = player.sort_values(
            ["season", "game_id"]
        ).copy()

        values = player[
            "shots_on_goal"
        ].fillna(0).to_numpy()

        seasons = player[
            "season"
        ].to_numpy()

        for line in [
            1.5,
            2.5,
            3.5
        ]:
            for index in range(
                1,
                len(player)
            ):
                current_season = seasons[
                    index
                ]

                under_streak = 0
                over_streak = 0

                for previous in range(
                    index - 1,
                    -1,
                    -1
                ):
                    if (
                        seasons[previous]
                        != current_season
                    ):
                        break

                    if (
                        values[previous]
                        < line
                    ):
                        under_streak += 1
                    else:
                        break

                for previous in range(
                    index - 1,
                    -1,
                    -1
                ):
                    if (
                        seasons[previous]
                        != current_season
                    ):
                        break

                    if (
                        values[previous]
                        > line
                    ):
                        over_streak += 1
                    else:
                        break

                if under_streak > 0:
                    rows.append({
                        "player_id": player_id,
                        "prop": (
                            f"Shots {line}"
                        ),
                        "type": "UNDER streak",
                        "streak": min(
                            under_streak,
                            MAX_STREAK
                        ),
                        "next_over": (
                            values[index]
                            > line
                        )
                    })

                if over_streak > 0:
                    rows.append({
                        "player_id": player_id,
                        "prop": (
                            f"Shots {line}"
                        ),
                        "type": "OVER streak",
                        "streak": min(
                            over_streak,
                            MAX_STREAK
                        ),
                        "next_over": (
                            values[index]
                            > line
                        )
                    })

    return pd.DataFrame(
        rows
    )


def print_binary_report(results):
    print()
    print(
        "POINT / GOAL / ASSIST DROUGHTS"
    )
    print("=" * 72)

    grouped = (
        results
        .groupby(
            [
                "prop",
                "streak"
            ]
        )
        .agg(
            samples=(
                "hit",
                "size"
            ),
            next_game_hits=(
                "hit",
                "sum"
            ),
            next_game_rate=(
                "hit",
                "mean"
            )
        )
        .reset_index()
    )

    grouped = grouped[
        grouped["samples"]
        >= MIN_SAMPLE
    ]

    for prop in [
        "Points",
        "Goals",
        "Assists"
    ]:
        print()
        print(prop)
        print("-" * 72)

        prop_rows = grouped[
            grouped["prop"] == prop
        ]

        for _, row in prop_rows.iterrows():
            label = (
                f"{int(row['streak'])}+"
                if row["streak"] == MAX_STREAK
                else str(
                    int(
                        row["streak"]
                    )
                )
            )

            print(
                f"{label}-game drought: "
                f"{int(row['next_game_hits']):,}/"
                f"{int(row['samples']):,} hit next game | "
                f"{row['next_game_rate']:.1%}"
            )


def print_shot_report(results):
    print()
    print()
    print(
        "SHOT STREAKS"
    )
    print("=" * 72)

    grouped = (
        results
        .groupby(
            [
                "prop",
                "type",
                "streak"
            ]
        )
        .agg(
            samples=(
                "next_over",
                "size"
            ),
            next_over=(
                "next_over",
                "sum"
            ),
            next_over_rate=(
                "next_over",
                "mean"
            )
        )
        .reset_index()
    )

    grouped = grouped[
        grouped["samples"]
        >= MIN_SAMPLE
    ]

    for prop in [
        "Shots 1.5",
        "Shots 2.5",
        "Shots 3.5"
    ]:
        print()
        print(prop)
        print("-" * 72)

        prop_rows = grouped[
            grouped["prop"] == prop
        ]

        for _, row in prop_rows.iterrows():
            label = (
                f"{int(row['streak'])}+"
                if row["streak"] == MAX_STREAK
                else str(
                    int(
                        row["streak"]
                    )
                )
            )

            print(
                f"{row['type']} "
                f"{label} games: "
                f"next game OVER "
                f"{row['next_over_rate']:.1%} | "
                f"{int(row['samples']):,} samples"
            )


def main():
    data = pd.read_csv(
        SKATER_FILE,
        dtype={
            "game_id": str,
            "player_id": str,
            "team_id": str
        }
    )

    data = data.dropna(
        subset=[
            "player_id",
            "season",
            "game_id"
        ]
    ).copy()

    binary_results = []

    for prop_name, target in PROPS.items():
        if target == "shots_on_goal":
            continue

        result = analyze_binary_prop(
            data,
            prop_name,
            target
        )

        binary_results.append(
            result
        )

    binary_results = pd.concat(
        binary_results,
        ignore_index=True
    )

    shot_results = analyze_shots(
        data
    )

    print()
    print(
        "NHL HISTORICAL STREAK REPORT"
    )
    print(
        "Tests what happened NEXT after "
        "specific player streaks/droughts."
    )

    print_binary_report(
        binary_results
    )

    print_shot_report(
        shot_results
    )


if __name__ == "__main__":
    main()