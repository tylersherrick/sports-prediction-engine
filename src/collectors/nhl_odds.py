
import json
import sys

import requests


SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/hockey/nhl/summary"
)


def implied_probability(odds):
    if odds is None:
        return None

    odds = float(odds)

    if odds < 0:
        return -odds / (-odds + 100)

    return 100 / (odds + 100)


def expected_value(probability, odds):
    if odds is None:
        return None

    odds = float(odds)

    profit = (
        odds / 100
        if odds > 0
        else 100 / abs(odds)
    )

    return probability * profit - (1 - probability)


def get_nhl_odds(game_id):
    response = requests.get(
        SUMMARY_URL,
        params={"event": game_id},
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()
    pickcenter = data.get("pickcenter", [])

    if not pickcenter:
        return None

    market = pickcenter[0]

    home_odds = market.get("homeTeamOdds") or {}
    away_odds = market.get("awayTeamOdds") or {}

    return {
        "provider": (
            market.get("provider") or {}
        ).get("name"),
        "moneyline": {
            "home": home_odds.get("moneyLine"),
            "away": away_odds.get("moneyLine"),
        },
        "puck_line": {
            "home": market.get("spread"),
            "home_odds": home_odds.get("spreadOdds"),
            "away_odds": away_odds.get("spreadOdds"),
        },
        "total": {
            "line": market.get("overUnder"),
            "over_odds": market.get("overOdds"),
            "under_odds": market.get("underOdds"),
        },
    }


def evaluate_game_lines(prediction, market):
    if not market:
        return {
            "puck_line": None,
            "total": None,
        }

    predicted_home_margin = prediction[
        "puck_line"
    ]["predicted_home_margin"]

    predicted_total = prediction[
        "total"
    ]["predicted_total"]

    market_spread = market[
        "puck_line"
    ]["home"]

    market_total = market[
        "total"
    ]["line"]

    puck_line = None
    total = None

    if market_spread is not None:
        margin_edge = (
            predicted_home_margin
            + float(market_spread)
        )

        puck_line = {
            "market_home_line": float(market_spread),
            "predicted_home_margin": predicted_home_margin,
            "lean": (
                "home"
                if margin_edge > 0
                else "away"
                if margin_edge < 0
                else "none"
            ),
            "margin_edge": round(margin_edge, 2),
            "home_odds": market["puck_line"]["home_odds"],
            "away_odds": market["puck_line"]["away_odds"],
        }

    if market_total is not None:
        total_edge = (
            predicted_total
            - float(market_total)
        )

        total = {
            "market_line": float(market_total),
            "predicted_total": predicted_total,
            "lean": (
                "over"
                if total_edge > 0
                else "under"
                if total_edge < 0
                else "none"
            ),
            "goal_edge": round(total_edge, 2),
            "over_odds": market["total"]["over_odds"],
            "under_odds": market["total"]["under_odds"],
        }

    return {
        "puck_line": puck_line,
        "total": total,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m src.collectors.nhl_odds GAME_ID"
        )

    print(
        json.dumps(
            get_nhl_odds(sys.argv[1]),
            indent=2,
        )
    )