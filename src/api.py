from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from src.nfl_prediction import get_nfl_prediction
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Sports Prediction Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
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
            <div class="game-date">{game_date_display}</div>
            <div class="game-id">ESPN Game ID: {result["game_id"]}</div>

            <div class="card">
                <div class="card-header">
                    <h2>WINNER / MONEYLINE</h2>
                    <div class="toggle">
                        <button id="odds-button" class="active" onclick="showOdds()">Odds</button>
                        <button id="percent-button" onclick="showPercent()">%</button>
                    </div>
                </div>

                <div class="pick">{winner["pick"]}</div>

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
                    <div class="pick">{value_pick} {value_market_odds}</div>

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
                <div class="pick">{total["model_total"]:.1f}</div>
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
                document.getElementById("odds-button").classList.add("active");
                document.getElementById("percent-button").classList.remove("active");
            }}

            function showPercent() {{
                document.querySelectorAll(".odds").forEach(
                    element => element.style.display = "none"
                );
                document.querySelectorAll(".percent").forEach(
                    element => element.style.display = "inline"
                );
                document.getElementById("percent-button").classList.add("active");
                document.getElementById("odds-button").classList.remove("active");
            }}
        </script>
    </body>
    </html>
    """