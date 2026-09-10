import joblib
import pandas as pd
import requests

GAMES_FILE = "data/processed/nfl_games.csv"
WINNER_MODEL = "models/nfl_2026_predictor.joblib"
SPREAD_MODEL = "models/nfl_2026_spread.joblib"
TOTAL_MODEL = "models/nfl_2026_total.joblib"

winner_saved = joblib.load(WINNER_MODEL)
spread_saved = joblib.load(SPREAD_MODEL)
total_saved = joblib.load(TOTAL_MODEL)

winner_model = winner_saved["model"]
winner_features = winner_saved["features"]
spread_model = spread_saved["model"]
spread_features = spread_saved["features"]
total_model = total_saved["model"]
total_features = total_saved["features"]

df = pd.read_csv(GAMES_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

def team_history(team, game_date):
    games = df[
        (
            (df["home_team"] == team) |
            (df["away_team"] == team)
        ) &
        (df["date"] < game_date)
    ].tail(5)

    history = []

    for _, game in games.iterrows():
        is_home = game["home_team"] == team

        if is_home:
            points = game["home_score"]
            points_allowed = game["away_score"]
            yards = game["home_total_yards"]
            yards_allowed = game["away_total_yards"]
            passing_yards = game["home_passing_yards"]
            turnovers = game["home_turnovers"]
            opponent_turnovers = game["away_turnovers"]
        else:
            points = game["away_score"]
            points_allowed = game["home_score"]
            yards = game["away_total_yards"]
            yards_allowed = game["home_total_yards"]
            passing_yards = game["away_passing_yards"]
            turnovers = game["away_turnovers"]
            opponent_turnovers = game["home_turnovers"]

        if points > points_allowed:
            result = 1
        elif points < points_allowed:
            result = 0
        else:
            result = 0.5

        history.append({
            "points": float(points),
            "points_allowed": float(points_allowed),
            "yards": float(yards),
            "yards_allowed": float(yards_allowed),
            "passing_yards": float(passing_yards),
            "turnovers": float(turnovers),
            "turnover_diff": float(opponent_turnovers) - float(turnovers),
            "result": result
        })

    return history

def average(history, field):
    return sum(game[field] for game in history) / len(history)

def american_odds(probability):
    if probability >= 0.5:
        return round(-100 * probability / (1 - probability))
    return round(100 * (1 - probability) / probability)

def implied_probability(odds):
    if odds < 0:
        return -odds / (-odds + 100)
    return 100 / (odds + 100)

def expected_value(probability, odds):
    if odds > 0:
        profit = odds / 100
    else:
        profit = 100 / abs(odds)

    return probability * profit - (1 - probability)

def get_nfl_prediction(game_id):
    response = requests.get(
        f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={game_id}",
        timeout=30
    )
    response.raise_for_status()
    data = response.json()

    competition = data["header"]["competitions"][0]
    competitors = competition["competitors"]
    game_date = pd.to_datetime(competition["date"])

    home_team = next(
        team["team"]["displayName"]
        for team in competitors
        if team["homeAway"] == "home"
    )
    away_team = next(
        team["team"]["displayName"]
        for team in competitors
        if team["homeAway"] == "away"
    )

    pickcenter = data.get("pickcenter", [])
    market = pickcenter[0] if pickcenter else {}

    market_spread = market.get("spread")
    market_total = market.get("overUnder")

    home_moneyline = market.get("homeTeamOdds", {}).get("moneyLine")
    away_moneyline = market.get("awayTeamOdds", {}).get("moneyLine")

    if home_moneyline is not None:
        home_moneyline = float(home_moneyline)

    if away_moneyline is not None:
        away_moneyline = float(away_moneyline)

    home = team_history(home_team, game_date)
    away = team_history(away_team, game_date)

    if not home or not away:
        raise ValueError("Not enough historical data for this matchup")

    row = {
        "home_avg_points_last_5": average(home, "points"),
        "away_avg_points_last_5": average(away, "points"),
        "home_avg_points_allowed_last_5": average(home, "points_allowed"),
        "away_avg_points_allowed_last_5": average(away, "points_allowed"),
        "home_avg_yards_last_5": average(home, "yards"),
        "away_avg_yards_last_5": average(away, "yards"),
        "home_avg_yards_allowed_last_5": average(home, "yards_allowed"),
        "away_avg_yards_allowed_last_5": average(away, "yards_allowed"),
        "home_avg_passing_yards_last_5": average(home, "passing_yards"),
        "away_avg_passing_yards_last_5": average(away, "passing_yards"),
        "home_avg_turnovers_last_5": average(home, "turnovers"),
        "away_avg_turnovers_last_5": average(away, "turnovers"),
        "home_avg_turnover_diff_last_5": average(home, "turnover_diff"),
        "away_avg_turnover_diff_last_5": average(away, "turnover_diff"),
        "home_win_pct_last_5": average(home, "result"),
        "away_win_pct_last_5": average(away, "result")
    }

    row["home_yards_matchup"] = (
        row["home_avg_yards_last_5"] -
        row["away_avg_yards_allowed_last_5"]
    )
    row["away_yards_matchup"] = (
        row["away_avg_yards_last_5"] -
        row["home_avg_yards_allowed_last_5"]
    )
    row["home_points_matchup"] = (
        row["home_avg_points_last_5"] -
        row["away_avg_points_allowed_last_5"]
    )
    row["away_points_matchup"] = (
        row["away_avg_points_last_5"] -
        row["home_avg_points_allowed_last_5"]
    )

    X_winner = pd.DataFrame([row])[winner_features]
    home_probability = winner_model.predict_proba(X_winner)[0][1]
    away_probability = 1 - home_probability

    if home_probability >= 0.5:
        winner = home_team
        winner_probability = home_probability
        winner_market_odds = home_moneyline
    else:
        winner = away_team
        winner_probability = away_probability
        winner_market_odds = away_moneyline

    winner_result = {
        "pick": winner,
        "probability": round(float(winner_probability), 3),
        "fair_odds": american_odds(winner_probability),
        "market_odds": None,
        "market_breakeven_probability": None,
        "value_pick": None,
        "value_model_probability": None,
        "value_fair_odds": None,
        "value_market_odds": None,
        "value_breakeven_probability": None,
        "value_edge": None,
        "expected_value": None
    }

    if winner_market_odds is not None:
        winner_result["market_odds"] = round(winner_market_odds)
        winner_result["market_breakeven_probability"] = round(
            implied_probability(winner_market_odds), 3
        )

    if home_moneyline is not None and away_moneyline is not None:
        home_breakeven = implied_probability(home_moneyline)
        away_breakeven = implied_probability(away_moneyline)

        home_edge = home_probability - home_breakeven
        away_edge = away_probability - away_breakeven

        if home_edge >= away_edge:
            value_pick = home_team
            value_probability = home_probability
            value_market_odds = home_moneyline
            value_breakeven = home_breakeven
            value_edge = home_edge
        else:
            value_pick = away_team
            value_probability = away_probability
            value_market_odds = away_moneyline
            value_breakeven = away_breakeven
            value_edge = away_edge

        if value_edge > 0:
            winner_result["value_pick"] = value_pick
            winner_result["value_model_probability"] = round(
                float(value_probability), 3
            )
            winner_result["value_fair_odds"] = american_odds(
                value_probability
            )
            winner_result["value_market_odds"] = round(
                value_market_odds
            )
            winner_result["value_breakeven_probability"] = round(
                value_breakeven, 3
            )
            winner_result["value_edge"] = round(
                float(value_edge), 3
            )
            winner_result["expected_value"] = round(
                expected_value(
                    value_probability,
                    value_market_odds
                ),
                3
            )

    X_spread = pd.DataFrame([row])[spread_features]
    predicted_margin = spread_model.predict(X_spread)[0]

    if predicted_margin >= 0:
        model_spread_team = home_team
        model_spread = -predicted_margin
    else:
        model_spread_team = away_team
        model_spread = predicted_margin

    X_total = pd.DataFrame([row])[total_features]
    predicted_total = total_model.predict(X_total)[0]

    spread_result = {
        "model_team": model_spread_team,
        "model_line": round(float(model_spread), 1),
        "market_team": None,
        "market_line": None,
        "edge_team": None,
        "edge": None
    }

    if market_spread is not None:
        market_spread = float(market_spread)

        if market_spread < 0:
            market_team = home_team
            market_line = market_spread
        elif market_spread > 0:
            market_team = away_team
            market_line = -market_spread
        else:
            market_team = home_team
            market_line = 0.0

        market_home_margin = -market_spread
        spread_edge = predicted_margin - market_home_margin

        spread_result["market_team"] = market_team
        spread_result["market_line"] = round(market_line, 1)
        spread_result["edge_team"] = (
            home_team if spread_edge >= 0 else away_team
        )
        spread_result["edge"] = round(abs(float(spread_edge)), 1)

    total_result = {
        "model_total": round(float(predicted_total), 1),
        "market_total": None,
        "lean": None,
        "edge": None
    }

    if market_total is not None:
        market_total = float(market_total)
        total_edge = predicted_total - market_total

        total_result["market_total"] = round(market_total, 1)
        total_result["lean"] = "over" if total_edge >= 0 else "under"
        total_result["edge"] = round(abs(float(total_edge)), 1)

    return {
        "game_id": str(game_id),
        "game_date": game_date.isoformat(),
        "away_team": away_team,
        "home_team": home_team,
        "winner": winner_result,
        "spread": spread_result,
        "total": total_result
    }