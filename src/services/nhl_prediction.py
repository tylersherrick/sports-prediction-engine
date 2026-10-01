from pathlib import Path

import joblib
import pandas as pd

FEATURES_FILE = Path("data/processed/nhl_features.csv")
MODEL_FILE = Path("models/nhl_winner.joblib")


class NHLPredictionService:
    def __init__(self):
        saved = joblib.load(MODEL_FILE)

        self.model = saved["model"]
        self.features = saved["features"]

    def get_game(self, game_id):
        data = pd.read_csv(
            FEATURES_FILE,
            dtype={
                "game_id": str,
                "home_team_id": str,
                "away_team_id": str
            }
        )

        game = data[
            data["game_id"] == str(game_id)
        ]

        if game.empty:
            return None

        return game.iloc[0]

    def predict_game(self, game_id):
        game = self.get_game(game_id)

        if game is None:
            return None

        X = pd.DataFrame(
            [game[self.features]],
            columns=self.features
        )

        home_probability = float(
            self.model.predict_proba(X)[0][1]
        )

        away_probability = 1 - home_probability

        if home_probability >= 0.5:
            predicted_winner = game["home_team"]
            win_probability = home_probability
        else:
            predicted_winner = game["away_team"]
            win_probability = away_probability

        return {
            "game_id": str(game["game_id"]),
            "away_team": game["away_team"],
            "home_team": game["home_team"],
            "winner": {
                "prediction": predicted_winner,
                "probability": round(
                    win_probability,
                    4
                ),
                "home_probability": round(
                    home_probability,
                    4
                ),
                "away_probability": round(
                    away_probability,
                    4
                )
            }
        }


nhl_prediction_service = NHLPredictionService()