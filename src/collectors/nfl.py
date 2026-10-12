import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/football/nfl/scoreboard"
)

SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/football/nfl/summary"
)

CURRENT_SEASON = 2026
BATCH_SIZE = 5


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


def get_games(week):
    return get_json(
        SCOREBOARD_URL,
        {
            "dates": CURRENT_SEASON,
            "seasontype": 2,
            "week": week
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
    competition = game[
        "competitions"
    ][0]

    return competition[
        "status"
    ][
        "type"
    ].get(
        "completed",
        False
    )


def save_game(game):
    if not is_completed(game):
        return None

    game_id = game["id"]

    game_path = (
        f"data/raw/nfl/{game_id}.json"
    )

    details_path = (
        f"data/raw/nfl/"
        f"{game_id}_details.json"
    )

    if (
        os.path.exists(game_path)
        and os.path.exists(details_path)
    ):
        return None

    details = get_game_details(
        game_id
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

    return game["name"]


def main():
    os.makedirs(
        "data/raw/nfl",
        exist_ok=True
    )

    start_time = time.time()

    new_games = 0

    print()
    print(
        f"Updating NFL {CURRENT_SEASON}"
    )

    print(
        "===================="
    )

    for week in range(1, 19):
        data = get_games(
            week
        )

        games = data.get(
            "events",
            []
        )

        completed = [
            game
            for game in games
            if is_completed(game)
        ]

        for start in range(
            0,
            len(completed),
            BATCH_SIZE
        ):
            batch = completed[
                start:
                start + BATCH_SIZE
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
                    result = future.result()

                    if result:
                        new_games += 1

                        print(
                            f"Saved {result}"
                        )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print(
        f"New games: "
        f"{new_games}"
    )

    print(
        f"Time: "
        f"{elapsed:.1f} seconds"
    )


if __name__ == "__main__":
    main()