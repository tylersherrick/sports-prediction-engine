import json
from collections import Counter
from pathlib import Path

RAW_DIR = Path("data/raw/nhl")

EXPECTED_SEASONS = [
    "2022",
    "2023",
    "2024",
    "2025",
    "2026"
]

def load_json(path):
    with open(path) as file:
        return json.load(file)

def get_competition(game):
    competitions = game.get(
        "competitions",
        []
    )

    if not competitions:
        return None

    return competitions[0]

def get_status(game):
    status = game.get(
        "status",
        {}
    )

    status_type = status.get(
        "type",
        {}
    )

    return {
        "name": status_type.get("name"),
        "description": status_type.get("description"),
        "completed": status_type.get("completed", False)
    }

def validate_season(season):
    season_dir = RAW_DIR / season

    if not season_dir.exists():
        print(f"\n{season}: NOT FOUND")
        return

    game_files = sorted(
        path
        for path in season_dir.glob("*.json")
        if not path.name.endswith("_details.json")
    )

    detail_files = sorted(
        season_dir.glob("*_details.json")
    )

    game_ids = []
    duplicate_ids = []
    seen_ids = set()

    completed = 0
    incomplete = 0
    missing_details = 0
    missing_boxscore = 0
    missing_players = 0
    missing_plays = 0

    status_counts = Counter()
    dates = []
    teams = Counter()

    for game_file in game_files:
        game = load_json(
            game_file
        )

        game_id = str(
            game.get("id")
        )

        if game_id in seen_ids:
            duplicate_ids.append(
                game_id
            )
        else:
            seen_ids.add(
                game_id
            )

        game_ids.append(
            game_id
        )

        game_date = game.get(
            "date"
        )

        if game_date:
            dates.append(
                game_date
            )

        status = get_status(
            game
        )

        status_counts[
            status["description"]
        ] += 1

        if status["completed"]:
            completed += 1
        else:
            incomplete += 1

        competition = get_competition(
            game
        )

        if competition:
            for competitor in competition.get(
                "competitors",
                []
            ):
                team = competitor.get(
                    "team",
                    {}
                )

                team_name = team.get(
                    "displayName"
                )

                if team_name:
                    teams[
                        team_name
                    ] += 1

        details_file = (
            season_dir /
            f"{game_id}_details.json"
        )

        if not details_file.exists():
            missing_details += 1
            continue

        details = load_json(
            details_file
        )

        boxscore = details.get(
            "boxscore",
            {}
        )

        if not boxscore:
            missing_boxscore += 1

        if not boxscore.get(
            "players"
        ):
            missing_players += 1

        if not details.get(
            "plays"
        ):
            missing_plays += 1

    orphan_details = 0

    for detail_file in detail_files:
        game_id = (
            detail_file.name
            .replace(
                "_details.json",
                ""
            )
        )

        if game_id not in seen_ids:
            orphan_details += 1

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"NHL {season}"
    )

    print(
        f"{'=' * 60}"
    )

    print(
        f"Game files: {len(game_files)}"
    )

    print(
        f"Detail files: {len(detail_files)}"
    )

    print(
        f"Unique game IDs: {len(seen_ids)}"
    )

    print(
        f"Duplicate IDs: {len(duplicate_ids)}"
    )

    print(
        f"Completed games: {completed}"
    )

    print(
        f"Incomplete games: {incomplete}"
    )

    print(
        f"Missing details: {missing_details}"
    )

    print(
        f"Orphan details: {orphan_details}"
    )

    print(
        f"Missing boxscores: {missing_boxscore}"
    )

    print(
        f"Missing player stats: {missing_players}"
    )

    print(
        f"Missing play-by-play: {missing_plays}"
    )

    print(
        f"Teams found: {len(teams)}"
    )

    if dates:
        print(
            f"First game: {min(dates)}"
        )

        print(
            f"Last game: {max(dates)}"
        )

    print(
        "\nStatuses:"
    )

    for status, count in sorted(
        status_counts.items()
    ):
        print(
            f"  {status}: {count}"
        )

    if duplicate_ids:
        print(
            "\nDuplicate game IDs:"
        )

        for game_id in duplicate_ids[
            :20
        ]:
            print(
                f"  {game_id}"
            )

def main():
    print(
        "Validating ESPN NHL raw data..."
    )

    for season in EXPECTED_SEASONS:
        validate_season(
            season
        )

    print(
        "\nValidation complete."
    )

if __name__ == "__main__":
    main()