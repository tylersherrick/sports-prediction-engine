from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from src.nfl_prediction import get_nfl_prediction
from src.nhl_prediction import get_nhl_prediction

app = FastAPI(title="Sports Prediction Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://tylersherrick.github.io",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Sports Prediction Engine running"}

@app.get("/nfl/{game_id}")
def nfl_prediction(game_id: str):
    try:
        return get_nfl_prediction(game_id)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))

@app.get("/nhl/{game_id}")
def nhl_prediction(game_id: str):
    try:
        return get_nhl_prediction(game_id)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))

def nhl_skater_rows(skaters):
    return "".join(
        f"""
        <tr>
            <td>
                <div class="player-name">{player["player"]}</div>
                <div class="player-meta">{player["team"]} · {player.get("position") or "-"}</div>
            </td>
            <td>{player["shots_on_goal"]:.2f}</td>
            <td>{player["points"]:.2f}</td>
            <td>{player["goals"]:.2f}</td>
            <td>{player["assists"]:.2f}</td>
        </tr>
        """
        for player in skaters
    )

def nhl_goalie_rows(goalies):
    return "".join(
        f"""
        <tr>
            <td>
                <div class="player-name">{goalie["player"]}</div>
                <div class="player-meta">{goalie["team"]}</div>
            </td>
            <td>{goalie["saves"]:.2f}</td>
        </tr>
        """
        for goalie in goalies
    )

@app.get("/nhl/{game_id}/report", response_class=HTMLResponse)
def nhl_report(game_id: str):
    try:
        result = get_nhl_prediction(game_id)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))

    winner = result["winner"]
    puck_line = result["puck_line"]
    total = result["total"]
    skaters = result.get("skaters", [])
    goalies = result.get("goalies", [])

    game_date = datetime.fromisoformat(result["game_date"])
    game_date = game_date.astimezone(ZoneInfo("America/Chicago"))
    game_date_display = game_date.strftime(
        "%A, %B %d, %Y • %I:%M %p CT"
    ).replace(" 0", " ")

    skater_rows = nhl_skater_rows(skaters)
    goalie_rows = nhl_goalie_rows(goalies)

    if not skater_rows:
        skater_rows = '<tr><td colspan="5" class="empty">No skater projections available.</td></tr>'

    if not goalie_rows:
        goalie_rows = '<tr><td colspan="2" class="empty">No goalie projections available.</td></tr>'

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{result["away_team"]} @ {result["home_team"]}</title>
        <style>
            * {{
                box-sizing: border-box;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: #111;
                color: #fff;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                max-width: 760px;
                margin: auto;
            }}
            h1 {{
                font-size: 24px;
                margin-bottom: 5px;
            }}
            .game-date {{
                color: #aaa;
                margin-bottom: 3px;
            }}
            .game-id {{
                color: #777;
                margin-bottom: 25px;
            }}
            .card {{
                background: #1c1c1e;
                border-radius: 14px;
                padding: 20px;
                margin-bottom: 15px;
            }}
            .card-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 14px;
            }}
            h2 {{
                font-size: 14px;
                color: #999;
                margin: 0 0 14px;
                letter-spacing: 1px;
            }}
            .card-header h2 {{
                margin: 0;
            }}
            .toggle {{
                display: flex;
                background: #2c2c2e;
                border-radius: 8px;
                padding: 2px;
            }}
            .toggle button {{
                border: 0;
                background: transparent;
                color: #999;
                padding: 5px 9px;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 600;
            }}
            .toggle button.active {{
                background: #555;
                color: #fff;
            }}
            .pick {{
                font-size: 22px;
                font-weight: 700;
                margin-bottom: 12px;
            }}
            .row {{
                display: flex;
                justify-content: space-between;
                gap: 20px;
                padding: 6px 0;
            }}
            .label {{
                color: #999;
            }}
            .value {{
                text-align: right;
                font-weight: 600;
            }}
            .nhl-percent {{
                display: none;
            }}
            .table-wrap {{
                overflow-x: auto;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
            }}
            th {{
                color: #777;
                font-size: 11px;
                letter-spacing: .7px;
                text-align: right;
                padding: 0 8px 10px;
            }}
            th:first-child {{
                text-align: left;
                padding-left: 0;
            }}
            td {{
                border-top: 1px solid #303033;
                padding: 11px 8px;
                text-align: right;
                font-weight: 600;
                white-space: nowrap;
            }}
            td:first-child {{
                text-align: left;
                padding-left: 0;
                white-space: normal;
            }}
            .player-name {{
                font-weight: 700;
            }}
            .player-meta {{
                color: #777;
                font-size: 12px;
                font-weight: 400;
                margin-top: 2px;
            }}
            .empty {{
                color: #777;
                text-align: left;
                font-weight: 400;
            }}
            .note {{
                color: #777;
                font-size: 12px;
                line-height: 1.4;
                margin-top: 12px;
            }}
            @media (max-width: 600px) {{
                body {{
                    padding: 14px;
                }}
                .card {{
                    padding: 16px;
                }}
                table {{
                    min-width: 560px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{result["away_team"]} @ {result["home_team"]}</h1>
            <div class="game-date">
                {game_date_display}
            </div>
            <div class="game-id">
                ESPN Game ID: {result["game_id"]}
            </div>

            <div class="card">
                <div class="card-header">
                    <h2>WINNER / MONEYLINE</h2>
                    <div class="toggle">
                        <button
                            id="nhl-odds-button"
                            class="active"
                            onclick="showNhlOdds()"
                        >
                            Odds
                        </button>
                        <button
                            id="nhl-percent-button"
                            onclick="showNhlPercent()"
                        >
                            %
                        </button>
                    </div>
                </div>

                <div class="pick">
                    {winner["pick"]}
                </div>

                <div class="row">
                    <span class="label">Model</span>
                    <span class="value nhl-odds">{winner["fair_odds"]:+d}</span>
                    <span class="value nhl-percent">{winner["probability"]:.1%}</span>
                </div>

                <div class="row">
                    <span class="label">{result["away_team"]}</span>
                    <span class="value">{winner["away_probability"]:.1%}</span>
                </div>

                <div class="row">
                    <span class="label">{result["home_team"]}</span>
                    <span class="value">{winner["home_probability"]:.1%}</span>
                </div>
            </div>

            <div class="card">
                <h2>PUCK LINE</h2>
                <div class="pick">
                    {puck_line["team"]} {puck_line["line"]:+.2f}
                </div>
                <div class="row">
                    <span class="label">Predicted home margin</span>
                    <span class="value">{puck_line["predicted_home_margin"]:+.2f}</span>
                </div>
            </div>

            <div class="card">
                <h2>TOTAL</h2>
                <div class="pick">
                    {total["predicted_total"]:.2f}
                </div>
                <div class="row">
                    <span class="label">Projected combined goals</span>
                    <span class="value">{total["predicted_total"]:.2f}</span>
                </div>
            </div>

            <div class="card">
                <h2>SKATER PROJECTIONS</h2>
                <div class="table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th>Player</th>
                                <th>SOG</th>
                                <th>PTS</th>
                                <th>G</th>
                                <th>A</th>
                            </tr>
                        </thead>
                        <tbody>
                            {skater_rows}
                        </tbody>
                    </table>
                </div>
                <div class="note">
                    These are model stat projections, not sportsbook lines or over/under probabilities.
                </div>
            </div>

            <div class="card">
                <h2>GOALIE SAVES</h2>
                <div class="table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th>Goalie</th>
                                <th>Saves</th>
                            </tr>
                        </thead>
                        <tbody>
                            {goalie_rows}
                        </tbody>
                    </table>
                </div>
                <div class="note">
                    Multiple goalies may appear until the expected starter is identified.
                </div>
            </div>
        </div>

        <script>
            function showNhlOdds() {{
                document.querySelectorAll(".nhl-odds").forEach(
                    element => element.style.display = "inline"
                );
                document.querySelectorAll(".nhl-percent").forEach(
                    element => element.style.display = "none"
                );
                document
                    .getElementById("nhl-odds-button")
                    .classList.add("active");
                document
                    .getElementById("nhl-percent-button")
                    .classList.remove("active");
            }}

            function showNhlPercent() {{
                document.querySelectorAll(".nhl-odds").forEach(
                    element => element.style.display = "none"
                );
                document.querySelectorAll(".nhl-percent").forEach(
                    element => element.style.display = "inline"
                );
                document
                    .getElementById("nhl-percent-button")
                    .classList.add("active");
                document
                    .getElementById("nhl-odds-button")
                    .classList.remove("active");
            }}
        </script>
    </body>
    </html>
    """

@app.get("/nfl/{game_id}/report", response_class=HTMLResponse)
def nfl_report(game_id: str):
    try:
        result = get_nfl_prediction(game_id)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))

    winner = result["winner"]
    spread = result["spread"]
    total = result["total"]

    game_date = datetime.fromisoformat(result["game_date"])
    game_date = game_date.astimezone(ZoneInfo("America/Chicago"))
    game_date_display = game_date.strftime(
        "%A, %B %d, %Y • %I:%M %p CT"
    ).replace(" 0", " ")

    winner_market_odds = (
        f'{winner["market_odds"]:+d}'
        if winner["market_odds"] is not None
        else "Unavailable"
    )

    winner_breakeven = (
        f'{winner["market_breakeven_probability"]:.1%}'
        if winner["market_breakeven_probability"] is not None
        else "Unavailable"
    )

    value_pick = (
        winner["value_pick"]
        if winner["value_pick"] is not None
        else "No positive value"
    )

    value_market_odds = (
        f'{winner["value_market_odds"]:+d}'
        if winner["value_market_odds"] is not None
        else "Unavailable"
    )

    value_model_probability = (
        f'{winner["value_model_probability"]:.1%}'
        if winner["value_model_probability"] is not None
        else "Unavailable"
    )

    value_breakeven = (
        f'{winner["value_breakeven_probability"]:.1%}'
        if winner["value_breakeven_probability"] is not None
        else "Unavailable"
    )

    value_edge = (
        f'+{winner["value_edge"]:.1%}'
        if winner["value_edge"] is not None
        else "Unavailable"
    )

    expected_value = (
        f'+{winner["expected_value"]:.1%}'
        if winner["expected_value"] is not None
        else "Unavailable"
    )

    spread_market = (
        f'{spread["market_team"]} {spread["market_line"]:+.1f}'
        if spread["market_line"] is not None
        else "Unavailable"
    )

    spread_edge = (
        f'{spread["edge_team"]} {spread["edge"]:.1f} points'
        if spread["edge"] is not None
        else "Unavailable"
    )

    total_market = (
        f'{total["market_total"]:.1f}'
        if total["market_total"] is not None
        else "Unavailable"
    )

    total_lean = (
        total["lean"].title()
        if total["lean"] is not None
        else "Unavailable"
    )

    total_edge = (
        f'{total["edge"]:.1f} points'
        if total["edge"] is not None
        else "Unavailable"
    )

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{result["away_team"]} @ {result["home_team"]}</title>

        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: #111;
                color: #fff;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                max-width: 500px;
                margin: auto;
            }}
            h1 {{
                font-size: 24px;
                margin-bottom: 5px;
            }}
            .game-date {{
                color: #aaa;
                margin-bottom: 3px;
            }}
            .game-id {{
                color: #777;
                margin-bottom: 25px;
            }}
            .card {{
                background: #1c1c1e;
                border-radius: 14px;
                padding: 20px;
                margin-bottom: 15px;
            }}
            .card-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 14px;
            }}
            h2 {{
                font-size: 14px;
                color: #999;
                margin: 0;
                letter-spacing: 1px;
            }}
            .toggle {{
                display: flex;
                background: #2c2c2e;
                border-radius: 8px;
                padding: 2px;
            }}
            .toggle button {{
                border: 0;
                background: transparent;
                color: #999;
                padding: 5px 9px;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 600;
            }}
            .toggle button.active {{
                background: #555;
                color: #fff;
            }}
            .pick {{
                font-size: 22px;
                font-weight: 700;
                margin-bottom: 12px;
            }}
            .row {{
                display: flex;
                justify-content: space-between;
                gap: 20px;
                padding: 6px 0;
            }}
            .label {{
                color: #999;
            }}
            .value {{
                text-align: right;
                font-weight: 600;
            }}
            .value-section {{
                border-top: 1px solid #333;
                margin-top: 16px;
                padding-top: 16px;
            }}
            .value-title {{
                font-size: 13px;
                color: #999;
                letter-spacing: 1px;
                margin-bottom: 8px;
            }}
            .percent {{
                display: none;
            }}
        </style>
    </head>

    <body>
        <div class="container">
            <h1>{result["away_team"]} @ {result["home_team"]}</h1>

            <div class="game-date">
                {game_date_display}
            </div>

            <div class="game-id">
                ESPN Game ID: {result["game_id"]}
            </div>

            <div class="card">
                <div class="card-header">
                    <h2>WINNER / MONEYLINE</h2>

                    <div class="toggle">
                        <button
                            id="odds-button"
                            class="active"
                            onclick="showOdds()"
                        >
                            Odds
                        </button>

                        <button
                            id="percent-button"
                            onclick="showPercent()"
                        >
                            %
                        </button>
                    </div>
                </div>

                <div class="pick">
                    {winner["pick"]}
                </div>

                <div class="row">
                    <span class="label">Model</span>
                    <span class="value odds">{winner["fair_odds"]:+d}</span>
                    <span class="value percent">{winner["probability"]:.1%}</span>
                </div>

                <div class="row">
                    <span class="label">Sportsbook</span>
                    <span class="value odds">{winner_market_odds}</span>
                    <span class="value percent">{winner_breakeven}</span>
                </div>

                <div class="value-section">
                    <div class="value-title">VALUE PICK</div>

                    <div class="pick">
                        {value_pick} {value_market_odds}
                    </div>

                    <div class="row">
                        <span class="label">Model probability</span>
                        <span class="value">{value_model_probability}</span>
                    </div>

                    <div class="row">
                        <span class="label">Breakeven probability</span>
                        <span class="value">{value_breakeven}</span>
                    </div>

                    <div class="row">
                        <span class="label">Betting edge</span>
                        <span class="value">{value_edge}</span>
                    </div>

                    <div class="row">
                        <span class="label">Expected value</span>
                        <span class="value">{expected_value}</span>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>SPREAD</h2>

                <div class="pick">
                    {spread["model_team"]} {spread["model_line"]:+.1f}
                </div>

                <div class="row">
                    <span class="label">Market</span>
                    <span class="value">{spread_market}</span>
                </div>

                <div class="row">
                    <span class="label">Model edge</span>
                    <span class="value">{spread_edge}</span>
                </div>
            </div>

            <div class="card">
                <h2>TOTAL</h2>

                <div class="pick">
                    {total["model_total"]:.1f}
                </div>

                <div class="row">
                    <span class="label">Market</span>
                    <span class="value">{total_market}</span>
                </div>

                <div class="row">
                    <span class="label">Lean</span>
                    <span class="value">{total_lean}</span>
                </div>

                <div class="row">
                    <span class="label">Model edge</span>
                    <span class="value">{total_edge}</span>
                </div>
            </div>
        </div>

        <script>
            function showOdds() {{
                document.querySelectorAll(".odds").forEach(
                    element => element.style.display = "inline"
                );

                document.querySelectorAll(".percent").forEach(
                    element => element.style.display = "none"
                );

                document
                    .getElementById("odds-button")
                    .classList.add("active");

                document
                    .getElementById("percent-button")
                    .classList.remove("active");
            }}

            function showPercent() {{
                document.querySelectorAll(".odds").forEach(
                    element => element.style.display = "none"
                );

                document.querySelectorAll(".percent").forEach(
                    element => element.style.display = "inline"
                );

                document
                    .getElementById("percent-button")
                    .classList.add("active");

                document
                    .getElementById("odds-button")
                    .classList.remove("active");
            }}
        </script>
    </body>
    </html>
    """