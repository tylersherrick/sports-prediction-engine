import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone

import requests


SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/hockey/nhl/scoreboard"
)

SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/hockey/nhl/summary"
)

SEASON = 2027
SEASON_START = date(2026, 9, 29)
BATCH_SIZE = 5

SEASON_PATH = f"data/raw/nhl/{SEASON}"


def get_json(url, params):
    for attempt in range(3):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=15
            )

            response.raise_for_status()

            return response.json()

        except requests.RequestException as error:
            if attempt == 2:
                raise error

            print(
                "Request failed. Retrying..."
            )

            time.sleep(2)


def get_games(game_date):
    return get_json(
        SCOREBOARD_URL,
        {
            "dates": game_date.strftime(
                "%Y%m%d"
            ),
            "limit": 100
        }
    )


def get_game_details(game_id):
    return get_json(
        SUMMARY_URL,
        {
            "event": game_id
        }
    )


def is_completed(game):
    return (
        game
        .get("status", {})
        .get("type", {})
        .get("completed", False)
    )


def get_game_date(game):
    game_date = game.get("date")

    if not game_date:
        return None

    return (
        datetime
        .fromisoformat(
            game_date.replace(
                "Z",
                "+00:00"
            )
        )
        .astimezone(timezone.utc)
        .date()
    )


def get_latest_saved_date():
    if not os.path.exists(
        SEASON_PATH
    ):
        return SEASON_START

    latest_date = None

    for filename in os.listdir(
        SEASON_PATH
    ):
        if not filename.endswith(
            ".json"
        ):
            continue

        if filename.endswith(
            "_details.json"
        ):
            continue

        path = os.path.join(
            SEASON_PATH,
            filename
        )

        try:
            with open(
                path,
                "r"
            ) as file:
                game = json.load(file)

        except (
            OSError,
            json.JSONDecodeError
        ):
            continue

        game_date = get_game_date(
            game
        )

        if game_date is None:
            continue

        if (
            latest_date is None
            or game_date > latest_date
        ):
            latest_date = game_date

    if latest_date is None:
        return SEASON_START

    return latest_date


def save_game(game):
    game_id = game["id"]

    game_path = os.path.join(
        SEASON_PATH,
        f"{game_id}.json"
    )

    details_path = os.path.join(
        SEASON_PATH,
        f"{game_id}_details.json"
    )

    if (
        os.path.exists(game_path)
        and os.path.exists(details_path)
    ):
        return (
            False,
            f"Already saved "
            f"{game['name']}"
        )

    details = get_game_details(
        game_id
    )

    boxscore = details.get(
        "boxscore",
        {}
    )

    if not boxscore.get(
        "players"
    ):
        raise RuntimeError(
            f"No player boxscore "
            f"for {game_id}"
        )

    with open(
        game_path,
        "w"
    ) as file:
        json.dump(
            game,
            file,
            indent=2
        )

    with open(
        details_path,
        "w"
    ) as file:
        json.dump(
            details,
            file,
            indent=2
        )

    return (
        True,
        f"Saved {game['name']}"
    )


def get_dates(
    start_date,
    end_date
):
    current_date = start_date

    while (
        current_date
        <= end_date
    ):
        yield current_date

        current_date += timedelta(
            days=1
        )


def main():
    os.makedirs(
        SEASON_PATH,
        exist_ok=True
    )

    start_time = time.time()

    start_date = (
        get_latest_saved_date()
        - timedelta(days=1)
    )

    if start_date < SEASON_START:
        start_date = SEASON_START

    end_date = date.today()

    print(
        "NHL Current-Season Update"
    )

    print(
        "========================="
    )

    print(
        f"Checking "
        f"{start_date} through "
        f"{end_date}"
    )

    new_games = 0

    for game_date in get_dates(
        start_date,
        end_date
    ):
        try:
            data = get_games(
                game_date
            )

        except Exception as error:
            print(
                f"ERROR scoreboard "
                f"{game_date}: {error}"
            )

            continue

        games = []

        for game in data.get(
            "events",
            []
        ):
            season_type = (
                game
                .get("season", {})
                .get("type")
            )

            if season_type != 2:
                continue

            if not is_completed(
                game
            ):
                continue

            games.append(game)

        if not games:
            continue

        print(
            f"{game_date}: "
            f"{len(games)} completed games"
        )

        for start in range(
            0,
            len(games),
            BATCH_SIZE
        ):
            batch = games[
                start:start + BATCH_SIZE
            ]

            with ThreadPoolExecutor(
                max_workers=BATCH_SIZE
            ) as executor:
                futures = [
                    executor.submit(
                        save_game,
                        game
                    )
                    for game in batch
                ]

                for future in as_completed(
                    futures
                ):
                    try:
                        saved, message = (
                            future.result()
                        )

                        print(message)

                        if saved:
                            new_games += 1

                    except Exception as error:
                        print(
                            f"ERROR: {error}"
                        )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print(
        "NHL update complete."
    )

    print(
        f"New games saved: "
        f"{new_games}"
    )

    print(
        f"Time: "
        f"{elapsed:.1f} seconds"
    )


if __name__ == "__main__":
    main()