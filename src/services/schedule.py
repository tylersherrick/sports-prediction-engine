from datetime import date

import requests


NFL_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/football/nfl/scoreboard"
)

NHL_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/hockey/nhl/scoreboard"
)


def get_scoreboard(url, game_date=None):
    if game_date is None:
        game_date = date.today()

    response = requests.get(
        url,
        params={
            "dates": game_date.strftime("%Y%m%d"),
            "limit": 100,
        },
        timeout=15,
    )

    response.raise_for_status()

    return response.json()


def get_nfl_games(game_date=None):
    data = get_scoreboard(
        NFL_SCOREBOARD_URL,
        game_date,
    )

    return data.get("events", [])


def get_nhl_games(game_date=None):
    data = get_scoreboard(
        NHL_SCOREBOARD_URL,
        game_date,
    )

    return data.get("events", [])