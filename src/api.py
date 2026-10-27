from datetime import date

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.collectors.nhl_odds import evaluate_game_lines
from src.nfl_prediction import get_nfl_prediction
from src.nhl_prediction import get_nhl_prediction
from src.picks import build_picks_page
from src.reports import build_nfl_report, build_nhl_report


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


@app.get("/picks", response_class=HTMLResponse)
def picks(
    selected_date: date | None = Query(
        default=None,
        alias="date",
    ),
):
    try:
        return build_picks_page(
            selected_date
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.get("/nfl/{game_id}")
def nfl_prediction(game_id: str):
    try:
        return get_nfl_prediction(game_id)
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@app.get("/nhl/{game_id}")
def nhl_prediction(game_id: str):
    try:
        prediction = get_nhl_prediction(game_id)

        prediction["market_comparison"] = evaluate_game_lines(
            prediction,
            prediction.get("market_odds"),
        )

        return prediction
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@app.get(
    "/nhl/{game_id}/report",
    response_class=HTMLResponse,
)
def nhl_report(game_id: str):
    try:
        result = get_nhl_prediction(game_id)

        result["market_comparison"] = evaluate_game_lines(
            result,
            result.get("market_odds"),
        )

        return build_nhl_report(result)
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@app.get(
    "/nfl/{game_id}/report",
    response_class=HTMLResponse,
)
def nfl_report(game_id: str):
    try:
        result = get_nfl_prediction(game_id)

        return build_nfl_report(result)
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )