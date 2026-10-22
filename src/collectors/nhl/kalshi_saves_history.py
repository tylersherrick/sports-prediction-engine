import csv
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


BASE_URL = (
    "https://external-api.kalshi.com/"
    "trade-api/v2"
)

SERIES_TICKER = "KXNHLSAVE"

OUTPUT_FILE = Path(
    "data/processed/"
    "nhl_kalshi_saves_history.csv"
)

EASTERN = ZoneInfo(
    "America/New_York"
)


def get_json(path, params=None):
    url = f"{BASE_URL}{path}"

    if params:
        url = (
            f"{url}?"
            f"{urlencode(params)}"
        )

    request = Request(
        url,
        headers={
            "User-Agent":
                "sports-prediction-engine/1.0",
            "Accept": "application/json"
        }
    )

    with urlopen(
        request,
        timeout=30
    ) as response:
        return json.loads(
            response
            .read()
            .decode("utf-8")
        )


def get_historical_cutoff():
    try:
        return get_json(
            "/historical/cutoff"
        )

    except Exception as error:
        print(
            "Could not fetch historical cutoff:"
        )
        print(error)

        return {}


def get_paginated_markets(
    path,
    params
):
    markets = []
    cursor = None

    while True:
        query = params.copy()

        query["limit"] = 1000

        if cursor:
            query["cursor"] = cursor

        data = get_json(
            path,
            query
        )

        page = data.get(
            "markets",
            []
        )

        markets.extend(page)

        print(
            f"{path}: "
            f"{len(markets):,} markets"
        )

        cursor = data.get(
            "cursor"
        )

        if not cursor:
            break

    return markets


def get_recent_settled_markets():
    return get_paginated_markets(
        "/markets",
        {
            "series_ticker":
                SERIES_TICKER,
            "status":
                "settled"
        }
    )


def get_archived_markets():
    return get_paginated_markets(
        "/historical/markets",
        {
            "series_ticker":
                SERIES_TICKER
        }
    )


def parse_market_title(title):
    if not title:
        return None, None

    match = re.match(
        r"^(.*?):\s*(\d+)\+\s+saves$",
        title.strip(),
        re.IGNORECASE
    )

    if not match:
        return None, None

    player = (
        match
        .group(1)
        .strip()
    )

    threshold = float(
        match.group(2)
    )

    return (
        player,
        threshold
    )


def parse_datetime(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value.replace(
                "Z",
                "+00:00"
            )
        )

    except (
        TypeError,
        ValueError
    ):
        return None


def get_game_datetime(market):
    candidates = [
        market.get(
            "occurrence_datetime"
        ),
        market.get(
            "expected_expiration_time"
        ),
        market.get(
            "close_time"
        ),
        market.get(
            "expiration_time"
        )
    ]

    for value in candidates:
        parsed = parse_datetime(
            value
        )

        if parsed is not None:
            return parsed

    return None


def get_game_date(market):
    game_datetime = (
        get_game_datetime(
            market
        )
    )

    if game_datetime is None:
        return None

    if game_datetime.tzinfo is None:
        game_datetime = (
            game_datetime.replace(
                tzinfo=ZoneInfo("UTC")
            )
        )

    eastern_time = (
        game_datetime.astimezone(
            EASTERN
        )
    )

    return (
        eastern_time
        .date()
        .isoformat()
    )


def get_price(
    market,
    side,
    price_type
):
    legacy_key = (
        f"{side}_{price_type}"
    )

    value = market.get(
        legacy_key
    )

    if value is not None:
        return value

    dollar_key = (
        f"{side}_"
        f"{price_type}_dollars"
    )

    value = market.get(
        dollar_key
    )

    if value is None:
        return None

    try:
        return round(
            float(value) * 100,
            2
        )

    except (
        TypeError,
        ValueError
    ):
        return None


def market_row(market):
    player, threshold = (
        parse_market_title(
            market.get(
                "title"
            )
        )
    )

    if (
        player is None
        or threshold is None
    ):
        return None

    game_date = get_game_date(
        market
    )

    return {
        "ticker":
            market.get(
                "ticker"
            ),

        "event_ticker":
            market.get(
                "event_ticker"
            ),

        "player":
            player,

        "threshold":
            threshold,

        "comparison_line":
            threshold - 0.5,

        "game_date":
            game_date,

        "status":
            market.get(
                "status"
            ),

        "result":
            market.get(
                "result"
            ),

        "occurrence_datetime":
            market.get(
                "occurrence_datetime"
            ),

        "open_time":
            market.get(
                "open_time"
            ),

        "close_time":
            market.get(
                "close_time"
            ),

        "expected_expiration_time":
            market.get(
                "expected_expiration_time"
            ),

        "expiration_time":
            market.get(
                "expiration_time"
            ),

        "settlement_ts":
            (
                market.get(
                    "settlement_ts"
                )
                or market.get(
                    "settled_at"
                )
            ),

        "yes_bid":
            get_price(
                market,
                "yes",
                "bid"
            ),

        "yes_ask":
            get_price(
                market,
                "yes",
                "ask"
            ),

        "no_bid":
            get_price(
                market,
                "no",
                "bid"
            ),

        "no_ask":
            get_price(
                market,
                "no",
                "ask"
            ),

        "last_price":
            (
                market.get(
                    "last_price"
                )
                or get_price(
                    market,
                    "last",
                    "price"
                )
            ),

        "volume":
            (
                market.get(
                    "volume"
                )
                or market.get(
                    "volume_fp"
                )
            ),

        "open_interest":
            (
                market.get(
                    "open_interest"
                )
                or market.get(
                    "open_interest_fp"
                )
            )
    }


def deduplicate(markets):
    unique = {}

    for market in markets:
        ticker = market.get(
            "ticker"
        )

        if not ticker:
            continue

        unique[ticker] = market

    return list(
        unique.values()
    )


def save_history(rows):
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    rows.sort(
        key=lambda row: (
            row["game_date"] or "",
            row["player"],
            row["threshold"]
        )
    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        if not rows:
            return

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def main():
    print()
    print(
        "KALSHI NHL SAVES HISTORY"
    )
    print("=" * 60)

    print()
    print(
        "Checking historical cutoff..."
    )

    cutoff = (
        get_historical_cutoff()
    )

    if cutoff:
        print(
            json.dumps(
                cutoff,
                indent=2
            )
        )

    print()
    print(
        "Fetching recent settled markets..."
    )

    recent = (
        get_recent_settled_markets()
    )

    print()
    print(
        "Fetching archived markets..."
    )

    archived = (
        get_archived_markets()
    )

    markets = deduplicate(
        recent + archived
    )

    rows = []

    skipped_titles = 0
    missing_dates = 0

    for market in markets:
        row = market_row(
            market
        )

        if row is None:
            skipped_titles += 1
            continue

        if not row["game_date"]:
            missing_dates += 1

        rows.append(
            row
        )

    save_history(
        rows
    )

    players = {
        row["player"]
        for row in rows
    }

    dates = {
        row["game_date"]
        for row in rows
        if row["game_date"]
    }

    goalie_games = {
        (
            row["player"],
            row["game_date"]
        )
        for row in rows
        if row["game_date"]
    }

    print()
    print("=" * 60)

    print(
        f"Recent settled: "
        f"{len(recent):,}"
    )

    print(
        f"Archived: "
        f"{len(archived):,}"
    )

    print(
        f"Raw unique markets: "
        f"{len(markets):,}"
    )

    print(
        f"Parsed saves markets: "
        f"{len(rows):,}"
    )

    print(
        f"Unique goalie-games: "
        f"{len(goalie_games):,}"
    )

    print(
        f"Unique goalies: "
        f"{len(players):,}"
    )

    print(
        f"Unique game dates: "
        f"{len(dates):,}"
    )

    print(
        f"Skipped titles: "
        f"{skipped_titles:,}"
    )

    print(
        f"Missing game dates: "
        f"{missing_dates:,}"
    )

    print(
        f"Saved: "
        f"{OUTPUT_FILE}"
    )

    if rows:
        print()
        print(
            "First market:"
        )

        print(
            rows[0]["game_date"],
            rows[0]["player"],
            f"{rows[0]['threshold']:.0f}+",
            rows[0]["result"]
        )

        print()
        print(
            "Last market:"
        )

        print(
            rows[-1]["game_date"],
            rows[-1]["player"],
            f"{rows[-1]['threshold']:.0f}+",
            rows[-1]["result"]
        )


if __name__ == "__main__":
    main()