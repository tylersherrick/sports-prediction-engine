from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile
from io import BytesIO

import pandas as pd


SEASONS = [
    2022,
    2023,
    2024,
    2025,
    2026
]

BASE_URL = (
    "https://moneypuck.com/moneypuck/playerData/"
    "seasonSummary/{season}/regular/goalies.csv"
)

OUTPUT_FILE = Path(
    "data/processed/nhl_moneypuck_goalies.csv"
)


def download_season(season):
    url = BASE_URL.format(
        season=season
    )

    print(
        f"Downloading MoneyPuck goalie data: "
        f"{season}-{season + 1}"
    )

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "sports-prediction-engine"
            )
        }
    )

    with urlopen(
        request,
        timeout=60
    ) as response:
        data = response.read()

    frame = pd.read_csv(
        BytesIO(data)
    )

    frame["season"] = season

    return frame


def main():
    print()
    print("MONEYPUCK NHL GOALIE COLLECTOR")
    print("=" * 60)

    frames = []

    for season in SEASONS:
        try:
            frame = download_season(
                season
            )

            print(
                f"  Rows: {len(frame):,}"
            )

            frames.append(
                frame
            )

        except Exception as error:
            print(
                f"  FAILED: {error}"
            )

    if not frames:
        raise RuntimeError(
            "No MoneyPuck goalie data downloaded."
        )

    goalies = pd.concat(
        frames,
        ignore_index=True
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    goalies.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 60)

    print(
        f"Total rows: "
        f"{len(goalies):,}"
    )

    print(
        f"Columns: "
        f"{len(goalies.columns):,}"
    )

    print(
        f"Saved: "
        f"{OUTPUT_FILE}"
    )

    print()
    print("AVAILABLE COLUMNS")
    print("-" * 60)

    for column in goalies.columns:
        print(column)


if __name__ == "__main__":
    main()