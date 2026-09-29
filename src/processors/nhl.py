import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/nhl")
PROCESSED_DIR = Path("data/processed")

SEASONS = [
    "2016",
    "2017",
    "2018",
    "2019",
    "2020",
    "2021",
    "2022",
    "2023",
    "2024",
    "2025",
    "2026"
]

SKATER_STAT_NAMES = {
    "BS": "blocked_shots",
    "HT": "hits",
    "TK": "takeaways",
    "+/-": "plus_minus",
    "TOI": "time_on_ice",
    "PPTOI": "power_play_time_on_ice",
    "SHTOI": "shorthanded_time_on_ice",
    "ESTOI": "even_strength_time_on_ice",
    "SHFT": "shifts",
    "G": "goals",
    "YTDG": "season_goals",
    "A": "assists",
    "S": "shots_on_goal",
    "SM": "shots_missed",
    "FW": "faceoffs_won",
    "FL": "faceoffs_lost",
    "FO%": "faceoff_pct",
    "GV": "giveaways",
    "PN": "penalties",
    "PIM": "penalty_minutes"
}

GOALIE_STAT_NAMES = {
    "GA": "goals_against",
    "SA": "shots_against",
    "SOS": "shootout_saves",
    "SOSA": "shootout_attempts",
    "SV": "saves",
    "SV%": "save_pct",
    "ESSV": "even_strength_saves",
    "PPSV": "power_play_saves",
    "SHSV": "shorthanded_saves",
    "TOI": "time_on_ice",
    "YTDG": "season_goals",
    "PIM": "penalty_minutes"
}


def load_json(path):
    with open(path) as file:
        return json.load(file)


def to_number(value):
    if value is None:
        return None

    if isinstance(
        value,
        (int, float)
    ):
        return value

    value = str(value).strip()

    if value in {
        "",
        "--",
        "-"
    }:
        return None

    try:
        return int(value)

    except ValueError:
        try:
            return float(value)

        except ValueError:
            return value


def time_to_seconds(value):
    if not value:
        return None

    try:
        minutes, seconds = (
            str(value)
            .split(":")
        )

        return (
            int(minutes) * 60
            + int(seconds)
        )

    except ValueError:
        return None


def get_completed(game):
    return (
        game
        .get("status", {})
        .get("type", {})
        .get("completed", False)
    )


def get_competition(game):
    competitions = game.get(
        "competitions",
        []
    )

    if not competitions:
        return None

    return competitions[0]


def get_competitors(game):
    competition = get_competition(
        game
    )

    if not competition:
        return None, None

    home = None
    away = None

    for competitor in competition.get(
        "competitors",
        []
    ):
        if competitor.get(
            "homeAway"
        ) == "home":
            home = competitor

        elif competitor.get(
            "homeAway"
        ) == "away":
            away = competitor

    return home, away


def get_team_info(competitor):
    if not competitor:
        return {}

    team = competitor.get(
        "team",
        {}
    )

    return {
        "team_id": team.get("id"),
        "team_abbreviation": team.get(
            "abbreviation"
        ),
        "team_name": team.get(
            "displayName"
        ),
        "score": to_number(
            competitor.get("score")
        ),
        "winner": competitor.get(
            "winner",
            False
        )
    }


def parse_team_statistics(details):
    result = {}

    teams = (
        details
        .get("boxscore", {})
        .get("teams", [])
    )

    for team_data in teams:
        team = team_data.get(
            "team",
            {}
        )

        team_id = str(
            team.get("id")
        )

        stats = {}

        for stat in team_data.get(
            "statistics",
            []
        ):
            name = stat.get(
                "name"
            )

            value = stat.get(
                "value"
            )

            if value is None:
                value = stat.get(
                    "displayValue"
                )

            stats[name] = to_number(
                value
            )

        result[team_id] = stats

    return result


def add_team_stats(
    row,
    prefix,
    team_id,
    team_stats
):
    stats = team_stats.get(
        str(team_id),
        {}
    )

    for name, value in stats.items():
        row[
            f"{prefix}_{name}"
        ] = value


def process_game(
    season,
    game,
    details
):
    home, away = get_competitors(
        game
    )

    if not home or not away:
        return None

    home_info = get_team_info(
        home
    )

    away_info = get_team_info(
        away
    )

    home_score = home_info[
        "score"
    ]

    away_score = away_info[
        "score"
    ]

    if (
        home_score is None
        or away_score is None
    ):
        return None

    competition = get_competition(
        game
    )

    venue = (
        competition
        .get("venue", {})
        .get("fullName")
        if competition
        else None
    )

    row = {
        "game_id": str(
            game.get("id")
        ),
        "season": int(
            season
        ),
        "date": game.get(
            "date"
        ),
        "name": game.get(
            "name"
        ),
        "venue": venue,
        "home_team_id": home_info[
            "team_id"
        ],
        "home_team": home_info[
            "team_name"
        ],
        "home_abbreviation": home_info[
            "team_abbreviation"
        ],
        "away_team_id": away_info[
            "team_id"
        ],
        "away_team": away_info[
            "team_name"
        ],
        "away_abbreviation": away_info[
            "team_abbreviation"
        ],
        "home_score": home_score,
        "away_score": away_score,
        "total_goals": (
            home_score
            + away_score
        ),
        "home_goal_diff": (
            home_score
            - away_score
        ),
        "home_win": int(
            home_score
            > away_score
        )
    }

    team_stats = parse_team_statistics(
        details
    )

    add_team_stats(
        row,
        "home",
        home_info["team_id"],
        team_stats
    )

    add_team_stats(
        row,
        "away",
        away_info["team_id"],
        team_stats
    )

    return row


def parse_player_stats(
    labels,
    stats,
    mapping
):
    parsed = {}

    for label, value in zip(
        labels,
        stats
    ):
        name = mapping.get(
            label
        )

        if not name:
            continue

        if name.endswith(
            "time_on_ice"
        ):
            parsed[name] = (
                time_to_seconds(
                    value
                )
            )

        else:
            parsed[name] = (
                to_number(
                    value
                )
            )

    return parsed


def process_players(
    season,
    game,
    details
):
    skaters = []
    goalies = []

    game_id = str(
        game.get("id")
    )

    game_date = game.get(
        "date"
    )

    home, away = get_competitors(
        game
    )

    home_info = get_team_info(
        home
    )

    away_info = get_team_info(
        away
    )

    home_team_id = str(
        home_info.get(
            "team_id"
        )
    )

    player_teams = (
        details
        .get("boxscore", {})
        .get("players", [])
    )

    for team_data in player_teams:
        team = team_data.get(
            "team",
            {}
        )

        team_id = str(
            team.get("id")
        )

        team_name = team.get(
            "displayName"
        )

        team_abbreviation = team.get(
            "abbreviation"
        )

        is_home = int(
            team_id
            == home_team_id
        )

        opponent = (
            away_info
            if is_home
            else home_info
        )

        for group in team_data.get(
            "statistics",
            []
        ):
            group_name = group.get(
                "name"
            )

            if group_name == "skaters":
                continue

            labels = group.get(
                "labels",
                []
            )

            athletes = group.get(
                "athletes",
                []
            )

            for player_data in athletes:
                athlete = player_data.get(
                    "athlete",
                    {}
                )

                player_id = athlete.get(
                    "id"
                )

                if not player_id:
                    continue

                position = (
                    athlete
                    .get("position", {})
                    .get("abbreviation")
                )

                base_row = {
                    "game_id": game_id,
                    "season": int(
                        season
                    ),
                    "date": game_date,
                    "team_id": team_id,
                    "team": team_name,
                    "team_abbreviation": (
                        team_abbreviation
                    ),
                    "opponent_team_id": (
                        opponent.get(
                            "team_id"
                        )
                    ),
                    "opponent": (
                        opponent.get(
                            "team_name"
                        )
                    ),
                    "opponent_abbreviation": (
                        opponent.get(
                            "team_abbreviation"
                        )
                    ),
                    "home": is_home,
                    "player_id": str(
                        player_id
                    ),
                    "player": athlete.get(
                        "displayName"
                    ),
                    "position": position,
                    "jersey": athlete.get(
                        "jersey"
                    ),
                    "active": athlete.get(
                        "active"
                    ),
                    "scratched": athlete.get(
                        "scratched"
                    )
                }

                stats = player_data.get(
                    "stats",
                    []
                )

                if group_name == "goalies":
                    row = dict(
                        base_row
                    )

                    row.update(
                        parse_player_stats(
                            labels,
                            stats,
                            GOALIE_STAT_NAMES
                        )
                    )

                    goalies.append(
                        row
                    )

                elif group_name in {
                    "forwards",
                    "defenses"
                }:
                    row = dict(
                        base_row
                    )

                    row[
                        "player_group"
                    ] = group_name

                    row.update(
                        parse_player_stats(
                            labels,
                            stats,
                            SKATER_STAT_NAMES
                        )
                    )

                    row["points"] = (
                        (
                            row.get(
                                "goals"
                            )
                            or 0
                        )
                        +
                        (
                            row.get(
                                "assists"
                            )
                            or 0
                        )
                    )

                    skaters.append(
                        row
                    )

    return skaters, goalies


def main():
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    games = []
    skaters = []
    goalies = []

    skipped = 0
    processed = 0

    for season in SEASONS:
        season_dir = (
            RAW_DIR /
            season
        )

        game_files = sorted(
            path
            for path in season_dir.glob(
                "*.json"
            )
            if not path.name.endswith(
                "_details.json"
            )
        )

        print(
            f"\nProcessing NHL {season}..."
        )

        for index, game_file in enumerate(
            game_files,
            start=1
        ):
            game = load_json(
                game_file
            )

            if not get_completed(
                game
            ):
                skipped += 1
                continue

            game_id = str(
                game.get("id")
            )

            details_file = (
                season_dir /
                f"{game_id}_details.json"
            )

            if not details_file.exists():
                skipped += 1
                continue

            details = load_json(
                details_file
            )

            game_row = process_game(
                season,
                game,
                details
            )

            if not game_row:
                skipped += 1
                continue

            game_skaters, game_goalies = (
                process_players(
                    season,
                    game,
                    details
                )
            )

            games.append(
                game_row
            )

            skaters.extend(
                game_skaters
            )

            goalies.extend(
                game_goalies
            )

            processed += 1

            if (
                processed % 250
                == 0
            ):
                print(
                    f"Processed "
                    f"{processed} games..."
                )

    games_df = pd.DataFrame(
        games
    )

    skaters_df = pd.DataFrame(
        skaters
    )

    goalies_df = pd.DataFrame(
        goalies
    )

    games_df = games_df.sort_values(
        [
            "date",
            "game_id"
        ]
    )

    skaters_df = skaters_df.sort_values(
        [
            "date",
            "game_id",
            "team",
            "player"
        ]
    )

    goalies_df = goalies_df.sort_values(
        [
            "date",
            "game_id",
            "team",
            "player"
        ]
    )

    games_file = (
        PROCESSED_DIR /
        "nhl_games.csv"
    )

    skaters_file = (
        PROCESSED_DIR /
        "nhl_skaters.csv"
    )

    goalies_file = (
        PROCESSED_DIR /
        "nhl_goalies.csv"
    )

    games_df.to_csv(
        games_file,
        index=False
    )

    skaters_df.to_csv(
        skaters_file,
        index=False
    )

    goalies_df.to_csv(
        goalies_file,
        index=False
    )

    print(
        "\nNHL processing complete."
    )

    print(
        f"Games: {len(games_df)}"
    )

    print(
        f"Skater game rows: "
        f"{len(skaters_df)}"
    )

    print(
        f"Goalie game rows: "
        f"{len(goalies_df)}"
    )

    print(
        f"Skipped raw events: "
        f"{skipped}"
    )

    print(
        "\nSaved:"
    )

    print(
        games_file
    )

    print(
        skaters_file
    )

    print(
        goalies_file
    )


if __name__ == "__main__":
    main()