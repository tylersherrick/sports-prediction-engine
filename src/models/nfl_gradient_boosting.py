import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

INPUT_FILE = "data/processed/nfl_features.csv"

features = [
    "home_avg_points_last_5",
    "away_avg_points_last_5",
    "home_avg_yards_last_5",
    "away_avg_yards_last_5",
    "home_avg_yards_allowed_last_5",
    "away_avg_yards_allowed_last_5",
    "home_avg_passing_yards_last_5",
    "away_avg_passing_yards_last_5",
    "home_avg_turnovers_last_5",
    "away_avg_turnovers_last_5",
    "home_win_pct_last_5",
    "away_win_pct_last_5"
]

df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=features + ["home_win"])

train = df[df["season"].isin([2022, 2023, 2024])]
test = df[df["season"] == 2025]

X_train = train[features]
y_train = train["home_win"]
X_test = test[features]
y_test = test["home_win"]

model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=2,
    learning_rate=0.05,
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