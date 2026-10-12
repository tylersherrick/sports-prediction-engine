import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

INPUT_FILE = "data/processed/nfl_features.csv"
MODEL_FILE = "models/nfl_2026_predictor.joblib"

features = [
    "home_avg_points_last_5",
    "away_avg_points_last_5",
    "home_avg_points_allowed_last_5",
    "away_avg_points_allowed_last_5",
    "home_avg_yards_last_5",
    "away_avg_yards_last_5",
    "home_avg_yards_allowed_last_5",
    "away_avg_yards_allowed_last_5",
    "home_avg_passing_yards_last_5",
    "away_avg_passing_yards_last_5",
    "home_avg_turnovers_last_5",
    "away_avg_turnovers_last_5",
    "home_avg_turnover_diff_last_5",
    "away_avg_turnover_diff_last_5",
    "home_win_pct_last_5",
    "away_win_pct_last_5",
    "home_yards_matchup",
    "away_yards_matchup",
    "home_points_matchup",
    "away_points_matchup"
]

df = pd.read_csv(INPUT_FILE)

df["home_yards_matchup"] = (
    df["home_avg_yards_last_5"] - df["away_avg_yards_allowed_last_5"]
)
df["away_yards_matchup"] = (
    df["away_avg_yards_last_5"] - df["home_avg_yards_allowed_last_5"]
)
df["home_points_matchup"] = (
    df["home_avg_points_last_5"] - df["away_avg_points_allowed_last_5"]
)
df["away_points_matchup"] = (
    df["away_avg_points_last_5"] - df["home_avg_points_allowed_last_5"]
)

df = df.dropna(subset=features + ["home_win"])

train = df[df["season"].isin([2022, 2023, 2024])]
test = df[df["season"] == 2025].copy()

test_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)

test_model.fit(train[features], train["home_win"])

test["home_win_probability"] = test_model.predict_proba(test[features])[:, 1]
test["predicted_home_win"] = (test["home_win_probability"] >= 0.5).astype(int)
test["correct"] = test["predicted_home_win"] == test["home_win"]

test["prediction_probability"] = test[
    "home_win_probability"
].where(
    test["predicted_home_win"] == 1,
    1 - test["home_win_probability"]
)

test["confidence"] = pd.cut(
    test["prediction_probability"],
    bins=[0.50, 0.55, 0.60, 0.65, 0.70, 1.00],
    labels=["50-55%", "55-60%", "60-65%", "65-70%", "70%+"],
    include_lowest=True
)

test["side"] = test["predicted_home_win"].map({
    1: "HOME",
    0: "AWAY"
})

accuracy = accuracy_score(
    test["home_win"],
    test["predicted_home_win"]
)

print("\n2025 Holdout Test")
print("=================")
print(f"Training games: {len(train)}")
print(f"Test games: {len(test)}")
print(f"Accuracy: {accuracy:.1%}")

print("\nConfidence Breakdown")
print("====================")

summary = test.groupby(
    "confidence",
    observed=True
)["correct"].agg(["sum", "count", "mean"])

summary["mean"] = (summary["mean"] * 100).round(1)

print(summary)

high_confidence = test[
    test["prediction_probability"] >= 0.65
]

print("\n2025 65%+ By Side")
print("=================")

side_summary = high_confidence.groupby(
    "side"
)["correct"].agg(["sum", "count", "mean"])

side_summary["mean"] = (
    side_summary["mean"] * 100
).round(1)

print(side_summary)

production_train = df[
    df["season"].isin([2022, 2023, 2024, 2025])
]

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)

model.fit(
    production_train[features],
    production_train["home_win"]
)

os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": features
    },
    MODEL_FILE
)

print("\nProduction Model")
print("================")
print(f"Training games: {len(production_train)}")
print("Training seasons: 2022-2025")
print(f"Model saved: {MODEL_FILE}")