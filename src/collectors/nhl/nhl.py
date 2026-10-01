import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import requests

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary"

BATCH_SIZE = 5

SEASON = 2027

START_DATE = date(
    2026,
    9,
    29
)

END_DATE = date.today()


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


def save_game(game):
    game_id = game["id"]

    season_path = (
        f"data/raw/nhl/{SEASON}"
    )

    game_path = (
        f"{season_path}/"
        f"{game_id}.json"
    )

    details_path = (
        f"{season_path}/"
        f"{game_id}_details.json"
    )

    if (
        os.path.exists(game_path)
        and os.path.exists(details_path)
    ):
        return (
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
        f"Saved {game['name']}"
    )


def get_dates(
    start_date,
    end_date
):
    current_date = start_date

    while current_date <= end_date:
        yield current_date

        current_date += timedelta(
            days=1
        )


def collect_current_season():
    season_path = (
        f"data/raw/nhl/{SEASON}"
    )

    os.makedirs(
        season_path,
        exist_ok=True
    )

    total_games = 0
    seen_games = set()
    start_time = time.time()

    print(
        f"\nCollecting NHL "
        f"2026-2027 completed games..."
    )

    for game_date in get_dates(
        START_DATE,
        END_DATE
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
            game_id = game["id"]

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

            if game_id in seen_games:
                continue

            seen_games.add(
                game_id
            )

            games.append(
                game
            )

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
                        print(
                            future.result()
                        )

                    except Exception as error:
                        print(
                            f"ERROR: {error}"
                        )

        total_games += len(
            games
        )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print(
        f"Done. Found "
        f"{total_games} completed games."
    )

    print(
        f"Time: {elapsed:.1f} seconds"
    )


def main():
    os.makedirs(
        "data/raw/nhl",
        exist_ok=True
    )

    collect_current_season()


if __name__ == "__main__":
    main()