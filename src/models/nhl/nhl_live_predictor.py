from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd
import requests

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/hockey/nhl/scoreboard"
)

FEATURES_FILE = Path(
    "data/processed/nhl_features.csv"
)

WINNER_MODEL_FILE = Path(
    "models/nhl_winner.joblib"
)

PUCK_LINE_MODEL_FILE = Path(
    "models/nhl_puck_line.joblib"
)

TOTAL_MODEL_FILE = Path(
    "models/nhl_total.joblib"
)

LOG_FILE = Path(
    "data/processed/nhl_live_predictions.csv"
)


def load_model(path):
    saved = joblib.load(
        path
    )

    return (
        saved["model"],
        saved["features"]
    )


def get_games():
    today = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d")

    response = requests.get(
        SCOREBOARD_URL,
        params={
            "dates": today,
            "limit": 100
        },
        timeout=15
    )

    response.raise_for_status()

    return response.json().get(
        "events",
        []
    )


def get_teams(game):
    competition = game[
        "competitions"
    ][0]

    home = None
    away = None

    for competitor in competition[
        "competitors"
    ]:
        if competitor[
            "homeAway"
        ] == "home":
            home = competitor

        elif competitor[
            "homeAway"
        ] == "away":
            away = competitor

    return home, away


def get_latest_team_row(
    data,
    team_id
):
    team_id = str(
        team_id
    )

    home_rows = data[
        data[
            "home_team_id"
        ].astype(str) == team_id
    ].copy()

    away_rows = data[
        data[
            "away_team_id"
        ].astype(str) == team_id
    ].copy()

    rows = []

    for _, row in home_rows.iterrows():
        values = {
            "date": row["date"]
        }

        for column in data.columns:
            if column.startswith(
                "home_"
            ):
                values[
                    column[5:]
                ] = row[column]

        rows.append(
            values
        )

    for _, row in away_rows.iterrows():
        values = {
            "date": row["date"]
        }

        for column in data.columns:
            if column.startswith(
                "away_"
            ):
                values[
                    column[5:]
                ] = row[column]

        rows.append(
            values
        )

    if not rows:
        return None

    rows = pd.DataFrame(
        rows
    )

    rows["date"] = pd.to_datetime(
        rows["date"],
        utc=True
    )

    return (
        rows
        .sort_values("date")
        .iloc[-1]
    )


def build_matchup(
    features,
    home_row,
    away_row
):
    matchup = {}

    for feature in features:
        if feature.startswith(
            "home_"
        ):
            base = feature[5:]

            matchup[
                feature
            ] = home_row.get(
                base
            )

        elif feature.startswith(
            "away_"
        ):
            base = feature[5:]

            matchup[
                feature
            ] = away_row.get(
                base
            )

        elif feature.endswith(
            "_diff"
        ):
            base = feature[
                :-5
            ]

            home_value = (
                home_row.get(
                    base
                )
            )

            away_value = (
                away_row.get(
                    base
                )
            )

            if (
                pd.notna(home_value)
                and pd.notna(away_value)
            ):
                matchup[
                    feature
                ] = (
                    home_value
                    - away_value
                )

            else:
                matchup[
                    feature
                ] = None

        else:
            matchup[
                feature
            ] = None

    return pd.DataFrame(
        [matchup],
        columns=features
    )


def load_existing_log():
    if not LOG_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(
        LOG_FILE,
        dtype={
            "game_id": str
        }
    )


def main():
    data = pd.read_csv(
        FEATURES_FILE,
        dtype={
            "game_id": str,
            "home_team_id": str,
            "away_team_id": str
        }
    )

    (
        winner_model,
        winner_features
    ) = load_model(
        WINNER_MODEL_FILE
    )

    (
        puck_line_model,
        puck_line_features
    ) = load_model(
        PUCK_LINE_MODEL_FILE
    )

    (
        total_model,
        total_features
    ) = load_model(
        TOTAL_MODEL_FILE
    )

    games = get_games()

    existing = load_existing_log()

    existing_ids = set()

    if not existing.empty:
        existing_ids = set(
            existing[
                "game_id"
            ].astype(str)
        )

    new_predictions = []

    print()
    print(
        "NHL Live Predictions"
    )

    print(
        "===================="
    )

    for game in games:
        game_id = str(
            game["id"]
        )

        season_type = (
            game
            .get("season", {})
            .get("type")
        )

        if season_type != 2:
            continue

        status = (
            game
            .get("status", {})
            .get("type", {})
        )

        if status.get(
            "completed",
            False
        ):
            continue

        if game_id in existing_ids:
            print()
            print(
                f"{game['name']}"
            )

            print(
                "Already logged."
            )

            continue

        home, away = get_teams(
            game
        )

        if not home or not away:
            continue

        home_team = home[
            "team"
        ]

        away_team = away[
            "team"
        ]

        home_id = str(
            home_team["id"]
        )

        away_id = str(
            away_team["id"]
        )

        home_row = get_latest_team_row(
            data,
            home_id
        )

        away_row = get_latest_team_row(
            data,
            away_id
        )

        if (
            home_row is None
            or away_row is None
        ):
            print()
            print(
                f"{away_team['displayName']} "
                f"@ {home_team['displayName']}"
            )

            print(
                "Missing team history."
            )

            continue

        winner_matchup = build_matchup(
            winner_features,
            home_row,
            away_row
        )

        home_probability = float(
            winner_model.predict_proba(
                winner_matchup
            )[0][1]
        )

        away_probability = (
            1
            - home_probability
        )

        if (
            home_probability
            >= 0.5
        ):
            predicted_winner = (
                home_team[
                    "displayName"
                ]
            )

            confidence = (
                home_probability
            )

        else:
            predicted_winner = (
                away_team[
                    "displayName"
                ]
            )

            confidence = (
                away_probability
            )

        puck_line_matchup = build_matchup(
            puck_line_features,
            home_row,
            away_row
        )

        predicted_home_margin = float(
            puck_line_model.predict(
                puck_line_matchup
            )[0]
        )

        if predicted_home_margin >= 0:
            puck_line_team = (
                home_team[
                    "displayName"
                ]
            )

            puck_line_value = (
                -predicted_home_margin
            )

        else:
            puck_line_team = (
                away_team[
                    "displayName"
                ]
            )

            puck_line_value = (
                predicted_home_margin
            )

        total_matchup = build_matchup(
            total_features,
            home_row,
            away_row
        )

        predicted_total = float(
            total_model.predict(
                total_matchup
            )[0]
        )

        print()
        print(
            f"{away_team['displayName']} "
            f"@ {home_team['displayName']}"
        )

        print(
            f"Winner: "
            f"{predicted_winner} "
            f"({confidence:.1%})"
        )

        print(
            f"Home probability: "
            f"{home_probability:.1%}"
        )

        print(
            f"Away probability: "
            f"{away_probability:.1%}"
        )

        print(
            f"Puck line: "
            f"{puck_line_team} "
            f"{puck_line_value:+.2f}"
        )

        print(
            f"Predicted home margin: "
            f"{predicted_home_margin:+.2f}"
        )

        print(
            f"Predicted total: "
            f"{predicted_total:.2f}"
        )

        new_predictions.append({
            "game_id": game_id,
            "game_date": game.get(
                "date"
            ),
            "logged_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "away_team_id": away_id,
            "away_team": away_team[
                "displayName"
            ],
            "home_team_id": home_id,
            "home_team": home_team[
                "displayName"
            ],
            "away_probability": (
                away_probability
            ),
            "home_probability": (
                home_probability
            ),
            "predicted_winner": (
                predicted_winner
            ),
            "puck_line_team": (
                puck_line_team
            ),
            "puck_line": (
                predicted_home_margin
                if predicted_home_margin < 0
                else -predicted_home_margin
            ),
            "predicted_home_margin": (
                predicted_home_margin
            ),
            "predicted_total": (
                predicted_total
            )
        })

    if not new_predictions:
        print()
        print(
            "No new games to log."
        )

        return

    new_df = pd.DataFrame(
        new_predictions
    )

    if existing.empty:
        combined = new_df

    else:
        combined = pd.concat(
            [
                existing,
                new_df
            ],
            ignore_index=True
        )

    LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    combined.to_csv(
        LOG_FILE,
        index=False
    )

    print()
    print(
        f"Logged "
        f"{len(new_predictions)} "
        f"new predictions."
    )

    print(
        f"Saved: {LOG_FILE}"
    )


if __name__ == "__main__":
    main()