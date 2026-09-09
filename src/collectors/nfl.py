import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
BATCH_SIZE = 5

def get_json(url, params):
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            if attempt == 2:
                raise error
            print("Request failed. Retrying...")
            time.sleep(2)

def get_games(season, week):
    return get_json(
        SCOREBOARD_URL,
        {
            "dates": season,
            "seasontype": 2,
            "week": week
        }
    )

def get_game_details(game_id):
    return get_json(
        SUMMARY_URL,
        {"event": game_id}
    )

def save_game(game):
    game_id = game["id"]
    game_path = f"data/raw/nfl/{game_id}.json"
    details_path = f"data/raw/nfl/{game_id}_details.json"

    if os.path.exists(game_path) and os.path.exists(details_path):
        return f"Already saved {game['name']}"

    details = get_game_details(game_id)

    with open(game_path, "w") as file:
        json.dump(game, file, indent=2)

    with open(details_path, "w") as file:
        json.dump(details, file, indent=2)

    return f"Saved {game['name']}"

os.makedirs("data/raw/nfl", exist_ok=True)

season = 2020
total_games = 0
start_time = time.time()

for week in range(1, 18):
    data = get_games(season, week)
    games = data["events"]
    print(f"Week {week}: {len(games)} games")

    for start in range(0, len(games), BATCH_SIZE):
        batch = games[start:start + BATCH_SIZE]

        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures = [executor.submit(save_game, game) for game in batch]

            for future in as_completed(futures):
                print(future.result())

    total_games += len(games)

elapsed = time.time() - start_time

print(f"Done. Found {total_games} games from the {season} regular season.")
print(f"Time: {elapsed:.1f} seconds")