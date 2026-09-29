import pandas as pd

GAME_RESULTS = [
    {
        "model": "Winner",
        "metric": "Accuracy",
        "model_score": 0.540,
        "baseline_score": 0.522,
        "better": "higher"
    },
    {
        "model": "Puck Line / Goal Margin",
        "metric": "MAE",
        "model_score": 2.133,
        "baseline_score": None,
        "better": "lower"
    },
    {
        "model": "Total Goals",
        "metric": "MAE",
        "model_score": 1.842,
        "baseline_score": 1.852,
        "better": "lower"
    }
]

PROP_RESULTS = [
    {
        "model": "Shots on Goal",
        "metric": "MAE",
        "model_score": 1.031,
        "baseline_score": 1.035,
        "better": "lower"
    },
    {
        "model": "Goalie Saves",
        "metric": "MAE",
        "model_score": 4.976,
        "baseline_score": 5.196,
        "better": "lower"
    },
    {
        "model": "Points",
        "metric": "MAE",
        "model_score": 0.527,
        "baseline_score": 0.528,
        "better": "lower"
    },
    {
        "model": "Goals",
        "metric": "MAE",
        "model_score": 0.271,
        "baseline_score": 0.270,
        "better": "lower"
    },
    {
        "model": "Assists",
        "metric": "MAE",
        "model_score": 0.404,
        "baseline_score": 0.403,
        "better": "lower"
    }
]

def evaluate_result(row):
    if pd.isna(row["baseline_score"]):
        return "NO BASELINE"

    if row["better"] == "higher":
        if row["model_score"] > row["baseline_score"]:
            return "BEATS BASELINE"

        if row["model_score"] < row["baseline_score"]:
            return "BELOW BASELINE"

    if row["better"] == "lower":
        if row["model_score"] < row["baseline_score"]:
            return "BEATS BASELINE"

        if row["model_score"] > row["baseline_score"]:
            return "BELOW BASELINE"

    return "TIED"

def calculate_change(row):
    if pd.isna(row["baseline_score"]):
        return None

    if row["better"] == "higher":
        return (
            row["model_score"]
            - row["baseline_score"]
        )

    return (
        row["baseline_score"]
        - row["model_score"]
    )

def print_section(title, results):
    df = pd.DataFrame(results)

    df["result"] = df.apply(
        evaluate_result,
        axis=1
    )

    df["improvement"] = df.apply(
        calculate_change,
        axis=1
    )

    display = df[
        [
            "model",
            "metric",
            "model_score",
            "baseline_score",
            "improvement",
            "result"
        ]
    ].copy()

    display.columns = [
        "Model",
        "Metric",
        "V1",
        "Baseline",
        "Improvement",
        "Result"
    ]

    print()
    print(title)
    print("=" * len(title))

    print(
        display.to_string(
            index=False,
            na_rep="-"
        )
    )

def main():
    print(
        "NHL MODEL BENCHMARK — V1"
    )

    print(
        "Test season: 2026"
    )

    print_section(
        "GAME MODELS",
        GAME_RESULTS
    )

    print_section(
        "PLAYER PROP MODELS",
        PROP_RESULTS
    )

    all_results = pd.DataFrame(
        GAME_RESULTS
        + PROP_RESULTS
    )

    comparable = all_results[
        all_results[
            "baseline_score"
        ].notna()
    ].copy()

    comparable["result"] = comparable.apply(
        evaluate_result,
        axis=1
    )

    beats = (
        comparable[
            "result"
        ]
        == "BEATS BASELINE"
    ).sum()

    below = (
        comparable[
            "result"
        ]
        == "BELOW BASELINE"
    ).sum()

    tied = (
        comparable[
            "result"
        ]
        == "TIED"
    ).sum()

    print()
    print("SUMMARY")
    print("=======")

    print(
        f"Models with comparable baselines: "
        f"{len(comparable)}"
    )

    print(
        f"Beat baseline: "
        f"{beats}"
    )

    print(
        f"Below baseline: "
        f"{below}"
    )

    print(
        f"Tied: "
        f"{tied}"
    )

if __name__ == "__main__":
    main()