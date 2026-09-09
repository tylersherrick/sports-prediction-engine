import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

INPUT_FILE = "data/processed/nfl_features.csv"

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
test = df[df["season"] == 2025]

X_train = train[features]
y_train = train["home_win"]
X_test = test[features]
y_test = test["home_win"]

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, predictions)

results = test[["week", "away_team", "home_team", "home_win"]].copy()
results["predicted_home_win"] = predictions
results["home_win_probability"] = probabilities
results["correct"] = results["home_win"] == results["predicted_home_win"]

print(f"Training games: {len(train)}")
print(f"Testing games: {len(test)}")
print(f"Accuracy: {accuracy:.3f}")
print(f"Correct: {results['correct'].sum()}/{len(results)}")