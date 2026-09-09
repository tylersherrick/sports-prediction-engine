import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss

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
    df["home_avg_yards_last_5"] -
    df["away_avg_yards_allowed_last_5"]
)

df["away_yards_matchup"] = (
    df["away_avg_yards_last_5"] -
    df["home_avg_yards_allowed_last_5"]
)

df["home_points_matchup"] = (
    df["home_avg_points_last_5"] -
    df["away_avg_points_allowed_last_5"]
)

df["away_points_matchup"] = (
    df["away_avg_points_last_5"] -
    df["home_avg_points_allowed_last_5"]
)

df = df.dropna(subset=features + ["home_win"])

train = df[df["season"].isin([2022, 2023, 2024])]
test = df[df["season"] == 2025]

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)

model.fit(train[features], train["home_win"])

home_probabilities = model.predict_proba(test[features])[:, 1]

results = test[["home_win"]].copy()
results["home_probability"] = home_probabilities
results["pick_probability"] = results["home_probability"].where(
    results["home_probability"] >= 0.5,
    1 - results["home_probability"]
)
results["pick_correct"] = (
    (results["home_probability"] >= 0.5) ==
    (results["home_win"] == 1)
)

bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.01]
labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75%+"]

results["confidence"] = pd.cut(
    results["pick_probability"],
    bins=bins,
    labels=labels,
    right=False
)

calibration = results.groupby(
    "confidence",
    observed=False
).agg(
    games=("pick_correct", "size"),
    correct=("pick_correct", "sum"),
    avg_prediction=("pick_probability", "mean"),
    actual_win_rate=("pick_correct", "mean")
)

print("\nNFL RANDOM FOREST CALIBRATION - 2025\n")

for confidence, row in calibration.iterrows():
    if row["games"] == 0:
        continue

    print(
        f"{confidence}: "
        f"{int(row['games'])} games | "
        f"avg prediction {row['avg_prediction']:.1%} | "
        f"actual {row['actual_win_rate']:.1%} | "
        f"{int(row['correct'])}/{int(row['games'])}"
    )

brier = brier_score_loss(
    results["home_win"],
    results["home_probability"]
)

print(f"\nBrier score: {brier:.3f}")
print("Lower Brier score = better probability accuracy.")