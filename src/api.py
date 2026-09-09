from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from src.nfl_prediction import get_nfl_prediction

app = FastAPI(title="Sports Prediction Engine")

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
            .game-id {{
                color: #888;
                margin-bottom: 25px;
            }}
            .card {{
                background: #1c1c1e;
                border-radius: 14px;
                padding: 20px;
                margin-bottom: 15px;
            }}
            h2 {{
                font-size: 14px;
                color: #999;
                margin: 0 0 14px;
                letter-spacing: 1px;
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{result["away_team"]} @ {result["home_team"]}</h1>
            <div class="game-id">ESPN Game ID: {result["game_id"]}</div>

            <div class="card">
                <h2>WINNER / MONEYLINE</h2>
                <div class="pick">{winner["pick"]}</div>
                <div class="row">
                    <span class="label">Probability</span>
                    <span class="value">{winner["probability"]:.1%}</span>
                </div>
                <div class="row">
                    <span class="label">Fair odds</span>
                    <span class="value">{winner["fair_odds"]:+d}</span>
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
    </body>
    </html>
    """