import re

import requests

from src.nhl_prediction import get_nhl_prediction
from src.collectors.nhl.propline_saves import (
    group_goalie_lines,
)


SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/hockey/nhl/scoreboard"
)

MIN_EDGE = -4.5
MAX_EDGE = -3.5


def normalize_name(name):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(name).lower(),
    )


def get_today_games():
    response = requests.get(
        SCOREBOARD_URL,
        timeout=30,
    )
    response.raise_for_status()

    games = []

    for event in response.json().get("events", []):
        games.append({
            "game_id": str(event["id"]),
            "name": event.get("name", ""),
        })

    return games


def load_draftkings():
    try:
        lines = group_goalie_lines()
    except Exception as error:
        print(f"DraftKings unavailable: {error}")
        return []

    return lines


def qualifies(edge):
    return (
        edge is not None
        and MIN_EDGE <= edge <= MAX_EDGE
    )


def get_kalshi_edge(goalie):
    market = goalie.get("kalshi")

    if not market:
        return None

    if market.get("pick") != "NO":
        return None

    edge = market.get("edge")

    if edge is None or not qualifies(edge):
        return None

    return {
        "source": "Kalshi",
        "edge": float(edge),
        "line": float(market["comparison_line"]),
        "display_line": market["threshold"],
        "price": market.get("price"),
        "ticker": market.get("ticker"),
        "pick": "NO",
    }


def get_draftkings_market(goalie, lines):
    goalie_key = normalize_name(goalie["player"])

    matches = [
        line
        for line in lines
        if normalize_name(line["goalie"]) == goalie_key
    ]

    if not matches:
        return None

    # Avoid guessing if the feed returns multiple possible lines.
    # Use the first available line for this goalie.
    market = matches[0]

    line = float(market["line"])
    projection = float(goalie["saves"])
    edge = round(projection - line, 2)

    if not qualifies(edge):
        return None

    return {
        "source": "DraftKings",
        "edge": edge,
        "line": line,
        "display_line": line,
        "price": market.get("under_price"),
        "ticker": None,
        "pick": "UNDER",
    }


def main():
    print()
    print("NHL GOALIE SAVES - TODAY")
    print("TARGET: NO / UNDER EDGE -3.5 TO -4.5")
    print("=" * 64)

    games = get_today_games()
    dk_lines = load_draftkings()

    print(f"Games scheduled: {len(games)}")
    print(f"DraftKings goalie lines loaded: {len(dk_lines)}")

    games_checked = 0
    games_failed = 0
    goalies_seen = 0
    kalshi_count = 0
    dk_count = 0
    missing_both = 0
    plays = []
    seen_goalies = set()

    for game in games:
        try:
            prediction = get_nhl_prediction(game["game_id"])
            games_checked += 1
        except Exception as error:
            games_failed += 1
            print(f"Game failed: {game['name']} — {error}")
            continue

        for goalie in prediction.get("goalies", []):
            goalies_seen += 1

            goalie_key = normalize_name(goalie["player"])
            seen_goalies.add(goalie_key)

            kalshi = goalie.get("kalshi")

            if kalshi:
                kalshi_count += 1
                market = get_kalshi_edge(goalie)

            else:
                market = get_draftkings_market(
                    goalie,
                    dk_lines,
                )

                if market:
                    dk_count += 1
                else:
                    missing_both += 1

            if not market:
                continue

            # Only keep NO/UNDER edges in our target range.
            if not qualifies(market["edge"]):
                continue

            plays.append({
                "game": (
                    f"{prediction['away_team']} "
                    f"@ {prediction['home_team']}"
                ),
                "goalie": goalie["player"],
                "team": goalie["team"],
                "projection": goalie["saves"],
                **market,
            })

    print()
    print("=" * 64)
    print("SCAN SUMMARY")
    print("=" * 64)
    print(f"Games scheduled:             {len(games)}")
    print(f"Games successfully checked:  {games_checked}")
    print(f"Games failed:                {games_failed}")
    print(f"Goalie entries evaluated:    {goalies_seen}")
    print(f"Entries with Kalshi markets: {kalshi_count}")
    print(f"Entries matched to DK:       {dk_count}")
    print(f"Entries missing both:        {missing_both}")
    print(f"Qualifying plays:            {len(plays)}")

    print()
    print("=" * 64)

    if not plays:
        print("NO QUALIFYING PLAYS")
        print("No matched market had an eligible NO/UNDER edge.")
        return

    print("QUALIFYING PLAYS")
    print("=" * 64)

    for play in sorted(
        plays,
        key=lambda item: item["edge"],
    ):
        print()
        print(play["game"])
        print(f"{play['goalie']} ({play['team']})")
        print(f"Source:     {play['source']}")
        print(f"Projection: {play['projection']:.2f} saves")
        print(f"Line:       {play['display_line']}")
        print(f"Edge:       {play['edge']:+.2f}")
        print(f"Pick:       {play['pick']}")
        print(f"Price:      {play['price']}")

        if play["ticker"]:
            print(f"Ticker:     {play['ticker']}")

    print()


if __name__ == "__main__":
    main()