import joblib
import pandas as pd
import requests


FEATURES_FILE = "data/processed/nhl_features.csv"
SKATER_FEATURES_FILE = "data/processed/nhl_skater_features.csv"
GOALIE_FEATURES_FILE = "data/processed/nhl_goalie_features.csv"

WINNER_MODEL = "models/nhl_winner.joblib"
PUCK_LINE_MODEL = "models/nhl_puck_line.joblib"
TOTAL_MODEL = "models/nhl_total.joblib"

SOG_MODEL = "models/nhl_sog.joblib"
POINTS_MODEL = "models/nhl_points.joblib"
GOALS_MODEL = "models/nhl_goals.joblib"
ASSISTS_MODEL = "models/nhl_assists.joblib"
SAVES_MODEL = "models/nhl_saves.joblib"

SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/hockey/nhl/summary"
)

PLAYER_RECENCY_DAYS = 120


def load_saved_model(path):
    saved = joblib.load(path)

    return (
        saved["model"],
        saved["features"]
    )


winner_model, winner_features = load_saved_model(
    WINNER_MODEL
)

puck_line_model, puck_line_features = load_saved_model(
    PUCK_LINE_MODEL
)

total_model, total_features = load_saved_model(
    TOTAL_MODEL
)

sog_model, sog_features = load_saved_model(
    SOG_MODEL
)

points_model, points_features = load_saved_model(
    POINTS_MODEL
)

goals_model, goals_features = load_saved_model(
    GOALS_MODEL
)

assists_model, assists_features = load_saved_model(
    ASSISTS_MODEL
)

saves_model, saves_features = load_saved_model(
    SAVES_MODEL
)


df = pd.read_csv(
    FEATURES_FILE,
    dtype={
        "game_id": str,
        "home_team_id": str,
        "away_team_id": str
    }
)

df["date"] = pd.to_datetime(
    df["date"],
    utc=True
)

df = df.sort_values(
    "date"
)


skater_df = pd.read_csv(
    SKATER_FEATURES_FILE,
    dtype={
        "game_id": str,
        "player_id": str,
        "team_id": str,
        "opponent_id": str
    }
)

skater_df["date"] = pd.to_datetime(
    skater_df["date"],
    utc=True
)

skater_df = skater_df.sort_values(
    "date"
)


goalie_df = pd.read_csv(
    GOALIE_FEATURES_FILE,
    dtype={
        "game_id": str,
        "player_id": str,
        "team_id": str,
        "opponent_id": str
    }
)

goalie_df["date"] = pd.to_datetime(
    goalie_df["date"],
    utc=True
)

goalie_df = goalie_df.sort_values(
    "date"
)


def american_odds(probability):
    if probability >= 0.5:
        return round(
            -100 * probability /
            (1 - probability)
        )

    return round(
        100 * (1 - probability) /
        probability
    )


def get_latest_team_row(team_id):
    team_id = str(team_id)

    home_games = df[
        df["home_team_id"] == team_id
    ]

    away_games = df[
        df["away_team_id"] == team_id
    ]

    rows = []

    for _, game in home_games.iterrows():
        row = {
            "date": game["date"]
        }

        for column in df.columns:
            if column.startswith("home_"):
                row[column[5:]] = game[column]

        rows.append(row)

    for _, game in away_games.iterrows():
        row = {
            "date": game["date"]
        }

        for column in df.columns:
            if column.startswith("away_"):
                row[column[5:]] = game[column]

        rows.append(row)

    if not rows:
        return None

    history = pd.DataFrame(rows)

    history = history.sort_values(
        "date"
    )

    return history.iloc[-1]


def build_matchup(
    home_row,
    away_row,
    features
):
    row = {}

    for feature in features:
        if feature.startswith("home_"):
            base = feature[5:]

            row[feature] = home_row.get(
                base
            )

        elif feature.startswith("away_"):
            base = feature[5:]

            row[feature] = away_row.get(
                base
            )

        elif feature.endswith("_diff"):
            base = feature[:-5]

            home_value = home_row.get(
                base
            )

            away_value = away_row.get(
                base
            )

            if (
                pd.notna(home_value)
                and pd.notna(away_value)
            ):
                row[feature] = (
                    home_value
                    - away_value
                )

            else:
                row[feature] = None

        else:
            row[feature] = None

    return pd.DataFrame(
        [row],
        columns=features
    )


def get_latest_player_rows(
    data,
    team_id
):
    team_id = str(team_id)

    team = data[
        data["team_id"] == team_id
    ].copy()

    if team.empty:
        return team

    team = team.sort_values(
        "date"
    )

    latest_team_date = team[
        "date"
    ].max()

    cutoff = (
        latest_team_date
        - pd.Timedelta(
            days=PLAYER_RECENCY_DAYS
        )
    )

    latest_players = (
        team
        .groupby(
            "player_id",
            as_index=False
        )
        .tail(1)
        .copy()
    )

    latest_players = latest_players[
        latest_players["date"] >= cutoff
    ].copy()

    return latest_players


def build_player_features(
    player_row,
    features,
    home
):
    row = {}

    for feature in features:
        if feature == "home":
            row[feature] = (
                1 if home else 0
            )

        else:
            row[feature] = (
                player_row.get(
                    feature
                )
            )

    return pd.DataFrame(
        [row],
        columns=features
    )


def get_skater_predictions(
    home_team_id,
    away_team_id
):
    predictions = []

    teams = [
        (
            home_team_id,
            True
        ),
        (
            away_team_id,
            False
        )
    ]

    for (
        team_id,
        home
    ) in teams:
        players = get_latest_player_rows(
            skater_df,
            team_id
        )

        for _, player in players.iterrows():
            if (
                player.get(
                    "player_games_played",
                    0
                ) < 5
            ):
                continue

            X_sog = build_player_features(
                player,
                sog_features,
                home
            )

            X_points = build_player_features(
                player,
                points_features,
                home
            )

            X_goals = build_player_features(
                player,
                goals_features,
                home
            )

            X_assists = build_player_features(
                player,
                assists_features,
                home
            )

            predicted_sog = float(
                sog_model.predict(
                    X_sog
                )[0]
            )

            predicted_points = float(
                points_model.predict(
                    X_points
                )[0]
            )

            predicted_goals = float(
                goals_model.predict(
                    X_goals
                )[0]
            )

            predicted_assists = float(
                assists_model.predict(
                    X_assists
                )[0]
            )

            predictions.append({
                "player_id": str(
                    player["player_id"]
                ),
                "player": player["player"],
                "team": player["team"],
                "position": player.get(
                    "position"
                ),
                "home": home,
                "shots_on_goal": round(
                    predicted_sog,
                    2
                ),
                "points": round(
                    predicted_points,
                    2
                ),
                "goals": round(
                    predicted_goals,
                    2
                ),
                "assists": round(
                    predicted_assists,
                    2
                )
            })

    predictions.sort(
        key=lambda item: (
            item["points"],
            item["shots_on_goal"]
        ),
        reverse=True
    )

    return predictions


def get_goalie_predictions(
    home_team_id,
    away_team_id
):
    predictions = []

    teams = [
        (
            home_team_id,
            True
        ),
        (
            away_team_id,
            False
        )
    ]

    for (
        team_id,
        home
    ) in teams:
        goalies = get_latest_player_rows(
            goalie_df,
            team_id
        )

        for _, goalie in goalies.iterrows():
            if (
                goalie.get(
                    "goalie_games_played",
                    0
                ) < 5
            ):
                continue

            X = build_player_features(
                goalie,
                saves_features,
                home
            )

            predicted_saves = float(
                saves_model.predict(
                    X
                )[0]
            )

            predictions.append({
                "player_id": str(
                    goalie["player_id"]
                ),
                "player": goalie["player"],
                "team": goalie["team"],
                "home": home,
                "saves": round(
                    predicted_saves,
                    2
                )
            })

    predictions.sort(
        key=lambda item: item["saves"],
        reverse=True
    )

    return predictions


def get_nhl_prediction(game_id):
    response = requests.get(
        SUMMARY_URL,
        params={
            "event": game_id
        },
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    competitions = (
        data
        .get(
            "header",
            {}
        )
        .get(
            "competitions",
            []
        )
    )

    if not competitions:
        raise ValueError(
            f"NHL game {game_id} not found"
        )

    competition = competitions[0]

    competitors = competition.get(
        "competitors",
        []
    )

    home = next(
        (
            team
            for team in competitors
            if team.get(
                "homeAway"
            ) == "home"
        ),
        None
    )

    away = next(
        (
            team
            for team in competitors
            if team.get(
                "homeAway"
            ) == "away"
        ),
        None
    )

    if (
        home is None
        or away is None
    ):
        raise ValueError(
            "Could not determine NHL teams"
        )

    home_team = home["team"][
        "displayName"
    ]

    away_team = away["team"][
        "displayName"
    ]

    home_team_id = str(
        home["team"]["id"]
    )

    away_team_id = str(
        away["team"]["id"]
    )

    game_date = pd.to_datetime(
        competition["date"],
        utc=True
    )

    home_row = get_latest_team_row(
        home_team_id
    )

    away_row = get_latest_team_row(
        away_team_id
    )

    if (
        home_row is None
        or away_row is None
    ):
        raise ValueError(
            "Not enough historical data "
            "for this matchup"
        )

    X_winner = build_matchup(
        home_row,
        away_row,
        winner_features
    )

    home_probability = float(
        winner_model
        .predict_proba(
            X_winner
        )[0][1]
    )

    away_probability = (
        1 - home_probability
    )

    if home_probability >= 0.5:
        winner = home_team
        winner_probability = (
            home_probability
        )

    else:
        winner = away_team
        winner_probability = (
            away_probability
        )

    winner_result = {
        "pick": winner,
        "probability": round(
            winner_probability,
            3
        ),
        "fair_odds": american_odds(
            winner_probability
        ),
        "home_probability": round(
            home_probability,
            3
        ),
        "away_probability": round(
            away_probability,
            3
        )
    }

    X_puck_line = build_matchup(
        home_row,
        away_row,
        puck_line_features
    )

    predicted_home_margin = float(
        puck_line_model.predict(
            X_puck_line
        )[0]
    )

    if predicted_home_margin >= 0:
        puck_line_team = (
            home_team
        )

        puck_line_value = (
            -predicted_home_margin
        )

    else:
        puck_line_team = (
            away_team
        )

        puck_line_value = (
            predicted_home_margin
        )

    puck_line_result = {
        "team": puck_line_team,
        "line": round(
            puck_line_value,
            2
        ),
        "predicted_home_margin": round(
            predicted_home_margin,
            2
        )
    }

    X_total = build_matchup(
        home_row,
        away_row,
        total_features
    )

    predicted_total = float(
        total_model.predict(
            X_total
        )[0]
    )

    total_result = {
        "predicted_total": round(
            predicted_total,
            2
        )
    }

    skaters = get_skater_predictions(
        home_team_id,
        away_team_id
    )

    goalies = get_goalie_predictions(
        home_team_id,
        away_team_id
    )

    return {
        "game_id": str(
            game_id
        ),
        "game_date": (
            game_date.isoformat()
        ),
        "away_team": away_team,
        "home_team": home_team,
        "winner": winner_result,
        "puck_line": puck_line_result,
        "total": total_result,
        "skaters": skaters,
        "goalies": goalies
    }