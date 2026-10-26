import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
SERIES_TICKER = "KXNHLSAVE"

OUTPUT_FILE = Path(
    "data/processed/nhl_kalshi_saves_markets.csv"
)

CACHE_SECONDS = 300

_saves_market_cache = None
_saves_market_cache_time = 0


def get_json(path, params=None):
    url = f"{BASE_URL}{path}"

    if params:
        url = f"{url}?{urlencode(params)}"

    request = Request(
        url,
        headers={
            "User-Agent": "sports-prediction-engine/1.0",
            "Accept": "application/json"
        }
    )

    with urlopen(
        request,
        timeout=30
    ) as response:
        return json.loads(
            response.read().decode("utf-8")
        )


def get_markets():
    markets = []
    cursor = None

    while True:
        params = {
            "series_ticker": SERIES_TICKER,
            "status": "open",
            "limit": 1000
        }

        if cursor:
            params["cursor"] = cursor

        data = get_json(
            "/markets",
            params
        )

        markets.extend(
            data.get(
                "markets",
                []
            )
        )

        cursor = data.get(
            "cursor"
        )

        if not cursor:
            break

    return markets


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

    player = match.group(
        1
    ).strip()

    threshold = float(
        match.group(2)
    )

    return player, threshold


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
        f"{side}_{price_type}_dollars"
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
    now = datetime.now(
        timezone.utc
    ).isoformat()

    title = market.get(
        "title"
    )

    player, threshold = (
        parse_market_title(
            title
        )
    )

    return {
        "captured_at": now,
        "ticker": market.get(
            "ticker"
        ),
        "event_ticker": market.get(
            "event_ticker"
        ),
        "player": player,
        "threshold": threshold,
        "title": title,
        "subtitle": market.get(
            "subtitle"
        ),
        "yes_sub_title": market.get(
            "yes_sub_title"
        ),
        "no_sub_title": market.get(
            "no_sub_title"
        ),
        "status": market.get(
            "status"
        ),
        "close_time": market.get(
            "close_time"
        ),
        "yes_bid": get_price(
            market,
            "yes",
            "bid"
        ),
        "yes_ask": get_price(
            market,
            "yes",
            "ask"
        ),
        "no_bid": get_price(
            market,
            "no",
            "bid"
        ),
        "no_ask": get_price(
            market,
            "no",
            "ask"
        ),
        "last_price": market.get(
            "last_price"
        ),
        "volume": market.get(
            "volume"
        ),
        "open_interest": market.get(
            "open_interest"
        ),
        "raw_market": json.dumps(
            market,
            separators=(",", ":")
        )
    }


def get_saves_markets(
    force_refresh=False
):
    global _saves_market_cache
    global _saves_market_cache_time

    now = time.monotonic()

    cache_valid = (
        _saves_market_cache
        is not None
        and (
            now
            - _saves_market_cache_time
        ) < CACHE_SECONDS
    )

    if (
        cache_valid
        and not force_refresh
    ):
        return _saves_market_cache

    markets = get_markets()

    rows = []

    for market in markets:
        row = market_row(
            market
        )

        if (
            row["player"] is None
            or row["threshold"] is None
        ):
            continue

        rows.append(
            row
        )

    _saves_market_cache = rows
    _saves_market_cache_time = now

    return rows


def normalize_name(name):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(name).lower()
    )


def find_goalie_market(
    player_name,
    markets=None
):
    if markets is None:
        markets = get_saves_markets()

    target = normalize_name(
        player_name
    )

    for market in markets:
        player = normalize_name(
            market["player"]
        )

        if player == target:
            return market

    return None


def save_markets(markets):
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not markets:
        print(
            "No open NHL saves markets found."
        )

        return

    file_exists = OUTPUT_FILE.exists()

    with OUTPUT_FILE.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=markets[
                0
            ].keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(
            markets
        )

    print()
    print(
        "KALSHI NHL SAVES"
    )
    print("=" * 60)

    print(
        f"Open markets found: "
        f"{len(markets)}"
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print()

    for market in markets:
        print(
            f"{market['player']} | "
            f"{market['threshold']:.0f}+ saves | "
            f"YES "
            f"{market['yes_bid']}/"
            f"{market['yes_ask']} | "
            f"NO "
            f"{market['no_bid']}/"
            f"{market['no_ask']}"
        )


def main():
    markets = get_saves_markets(
        force_refresh=True
    )

    save_markets(
        markets
    )


if __name__ == "__main__":
    main()