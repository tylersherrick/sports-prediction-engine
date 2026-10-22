import requests


BASE_URL = "https://api.prop-line.com/v1"
SPORT = "hockey_nhl"
MARKET = "goalie_saves"

API_KEY = "a937021cf3ee1e041bad79394785aa39"


def get_events():
    if not API_KEY:
        raise RuntimeError(
            "PROPLINE_API_KEY is not set."
        )

    response = requests.get(
        f"{BASE_URL}/sports/{SPORT}/events",
        headers={
            "X-API-Key": API_KEY
        },
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def get_event_saves(
    event_id
):
    response = requests.get(
        (
            f"{BASE_URL}/sports/"
            f"{SPORT}/events/"
            f"{event_id}/odds"
        ),
        params={
            "markets": MARKET,
            "bookmakers": "draftkings"
        },
        headers={
            "X-API-Key": API_KEY
        },
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def get_draftkings_saves():
    events = get_events()

    props = []

    for event in events:
        event_id = event.get(
            "id"
        )

        if not event_id:
            continue

        try:
            data = get_event_saves(
                event_id
            )

        except requests.RequestException:
            continue

        for bookmaker in data.get(
            "bookmakers",
            []
        ):
            if bookmaker.get(
                "key"
            ) != "draftkings":
                continue

            for market in bookmaker.get(
                "markets",
                []
            ):
                if market.get(
                    "key"
                ) != MARKET:
                    continue

                for outcome in market.get(
                    "outcomes",
                    []
                ):
                    player = outcome.get(
                        "description"
                    )

                    side = outcome.get(
                        "name"
                    )

                    point = outcome.get(
                        "point"
                    )

                    price = outcome.get(
                        "price"
                    )

                    if (
                        not player
                        or side not in {
                            "Over",
                            "Under"
                        }
                        or point is None
                    ):
                        continue

                    props.append({
                        "event_id": event_id,
                        "home_team": data.get(
                            "home_team"
                        ),
                        "away_team": data.get(
                            "away_team"
                        ),
                        "goalie": player,
                        "side": side,
                        "line": float(
                            point
                        ),
                        "price": price,
                        "book": "DraftKings"
                    })

    return props


def group_goalie_lines():
    props = get_draftkings_saves()

    goalies = {}

    for prop in props:
        key = (
            prop["event_id"],
            prop["goalie"],
            prop["line"]
        )

        if key not in goalies:
            goalies[key] = {
                "event_id": prop[
                    "event_id"
                ],
                "home_team": prop[
                    "home_team"
                ],
                "away_team": prop[
                    "away_team"
                ],
                "goalie": prop[
                    "goalie"
                ],
                "line": prop[
                    "line"
                ],
                "over_price": None,
                "under_price": None,
                "book": "DraftKings"
            }

        if prop["side"] == "Over":
            goalies[key][
                "over_price"
            ] = prop["price"]

        elif prop["side"] == "Under":
            goalies[key][
                "under_price"
            ] = prop["price"]

    return list(
        goalies.values()
    )


if __name__ == "__main__":
    lines = group_goalie_lines()

    print()
    print(
        "DRAFTKINGS GOALIE SAVES"
    )
    print("=" * 60)

    print(
        f"Goalie lines found: "
        f"{len(lines)}"
    )

    print("=" * 60)

    for line in lines:
        print()

        print(
            f"{line['away_team']} "
            f"@ {line['home_team']}"
        )

        print(
            line["goalie"]
        )

        print(
            f"Line: "
            f"{line['line']:.1f}"
        )

        print(
            f"Over: "
            f"{line['over_price']}"
        )

        print(
            f"Under: "
            f"{line['under_price']}"
        )

    print()