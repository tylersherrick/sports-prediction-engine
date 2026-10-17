
from src.collectors.nhl_odds import (
    expected_value,
    implied_probability,
)
from src.nhl_prediction import get_nhl_prediction


def evaluate_nhl_game(game_id):
    prediction = get_nhl_prediction(game_id)
    market = prediction.get("market_odds")

    if not market:
        return {
            "game_id": game_id,
            "error": "No sportsbook odds available",
        }

    home_probability = prediction["winner"]["home_probability"]
    away_probability = prediction["winner"]["away_probability"]

    home_odds = market["moneyline"]["home"]
    away_odds = market["moneyline"]["away"]

    results = []

    for team, probability, odds in [
        (prediction["home_team"], home_probability, home_odds),
        (prediction["away_team"], away_probability, away_odds),
    ]:
        if odds is None:
            continue

        breakeven = implied_probability(odds)
        edge = probability - breakeven
        ev = expected_value(probability, odds)

        results.append({
            "team": team,
            "model_probability": round(probability, 3),
            "market_odds": odds,
            "breakeven_probability": round(breakeven, 3),
            "edge": round(edge, 3),
            "expected_value": round(ev, 3),
        })

    results.sort(
        key=lambda item: item["expected_value"],
        reverse=True,
    )

    return {
        "game_id": str(game_id),
        "provider": market["provider"],
        "best_value": (
            results[0]
            if results and results[0]["expected_value"] > 0
            else None
        ),
        "moneyline": results,
    }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m src.evaluation.nhl_betting GAME_ID"
        )

    print(
        json.dumps(
            evaluate_nhl_game(sys.argv[1]),
            indent=2,
        )
    )
