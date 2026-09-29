import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import requests

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary"

BATCH_SIZE = 5

SEASONS = [
    {
        "season": 2016,
        "start": date(2015, 10, 7),
        "end": date(2016, 4, 10)
    },
    {
        "season": 2017,
        "start": date(2016, 10, 12),
        "end": date(2017, 4, 9)
    },
    {
        "season": 2018,
        "start": date(2017, 10, 4),
        "end": date(2018, 4, 8)
    },
    {
        "season": 2019,
        "start": date(2018, 10, 3),
        "end": date(2019, 4, 6)
    },
    {
        "season": 2020,
        "start": date(2019, 10, 2),
        "end": date(2020, 3, 11)
    },
    {
        "season": 2021,
        "start": date(2021, 1, 13),
        "end": date(2021, 5, 19)
    },
    {
        "season": 2022,
        "start": date(2021, 10, 12),
        "end": date(2022, 4, 29)
    },
    {
        "season": 2023,
        "start": date(2022, 10, 7),
        "end": date(2023, 4, 14)
    },
    {
        "season": 2024,
        "start": date(2023, 10, 10),
        "end": date(2024, 4, 18)
    },
    {
        "season": 2025,
        "start": date(2024, 10, 4),
        "end": date(2025, 4, 17)
    },
    {
        "season": 2026,
        "start": date(2025, 10, 7),
        "end": date(2026, 4, 16)
    }
]


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

            print("Request failed. Retrying...")

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


def save_game(game, season):
    game_id = game["id"]

    season_path = (
        f"data/raw/nhl/{season}"
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


def collect_season(
    season,
    start_date,
    end_date
):
    season_path = (
        f"data/raw/nhl/{season}"
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
        f"{season - 1}-{season}..."
    )

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
            game_id = game["id"]

            season_type = (
                game
                .get("season", {})
                .get("type")
            )

            if season_type != 2:
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
            f"{len(games)} games"
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
                        game,
                        season
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

    print(
        f"\nDone. Found "
        f"{total_games} games from "
        f"the {season - 1}-{season} "
        f"regular season."
    )

    print(
        f"Time: {elapsed:.1f} seconds"
    )


def main():
    os.makedirs(
        "data/raw/nhl",
        exist_ok=True
    )

    for season in SEASONS:
        collect_season(
            season["season"],
            season["start"],
            season["end"]
        )


if __name__ == "__main__":
    main()