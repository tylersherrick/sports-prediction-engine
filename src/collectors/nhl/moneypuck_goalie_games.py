from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import random
import time

import pandas as pd


SUMMARY_FILE = Path(
    "data/processed/nhl_moneypuck_goalies.csv"
)

OUTPUT_FILE = Path(
    "data/processed/nhl_moneypuck_goalie_games.csv"
)

BASE_URL = (
    "https://moneypuck.com/moneypuck/playerData/"
    "careers/gameByGame/regular/goalies/{player_id}.csv"
)

MIN_SEASON = 2022

REQUEST_DELAY_MIN = 1.0
REQUEST_DELAY_MAX = 2.0

MAX_RETRIES = 6

RETRY_DELAYS = [
    10,
    20,
    40,
    60,
    90,
    120
]


def download_goalie(
    player_id,
    name
):
    url = BASE_URL.format(
        player_id=player_id
    )

    for attempt in range(MAX_RETRIES + 1):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 "
                        "(Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 "
                        "(KHTML, like Gecko) "
                        "Chrome/155.0.0.0 "
                        "Safari/537.36"
                    )
                }
            )

            with urlopen(
                request,
                timeout=60
            ) as response:
                data = response.read()

            return pd.read_csv(
                BytesIO(data)
            )

        except HTTPError as error:
            if (
                error.code == 429
                and attempt < MAX_RETRIES
            ):
                wait_time = RETRY_DELAYS[
                    min(
                        attempt,
                        len(RETRY_DELAYS) - 1
                    )
                ]

                print(
                    f"  Rate limited. "
                    f"Waiting {wait_time}s..."
                )

                time.sleep(
                    wait_time
                )

                continue

            raise

        except Exception:
            if attempt < MAX_RETRIES:
                wait_time = RETRY_DELAYS[
                    min(
                        attempt,
                        len(RETRY_DELAYS) - 1
                    )
                ]

                print(
                    f"  Request failed. "
                    f"Retrying in {wait_time}s..."
                )

                time.sleep(
                    wait_time
                )

                continue

            raise

    raise RuntimeError(
        f"Failed to download {name}"
    )


def main():
    print()
    print("MONEYPUCK NHL GOALIE GAME COLLECTOR")
    print("=" * 60)

    summary = pd.read_csv(
        SUMMARY_FILE,
        dtype={
            "playerId": str
        }
    )

    summary["season"] = pd.to_numeric(
        summary["season"],
        errors="coerce"
    )

    summary = summary[
        summary["season"] >= MIN_SEASON
    ].copy()

    players = (
        summary[
            [
                "playerId",
                "name"
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=["playerId"]
        )
        .sort_values("name")
        .reset_index(drop=True)
    )

    print(
        f"Goalies to download: "
        f"{len(players):,}"
    )

    print()

    frames = []
    failed = []

    for index, row in players.iterrows():
        player_id = str(
            row["playerId"]
        )

        name = row["name"]

        print(
            f"[{index + 1}/{len(players)}] "
            f"{name}"
        )

        try:
            frame = download_goalie(
                player_id,
                name
            )

            frame["playerId"] = (
                frame["playerId"]
                .astype(str)
            )

            frame["season"] = pd.to_numeric(
                frame["season"],
                errors="coerce"
            )

            frame = frame[
                frame["season"] >= MIN_SEASON
            ].copy()

            if frame.empty:
                print(
                    "  No games since 2022"
                )

            else:
                print(
                    f"  Games: {len(frame):,}"
                )

                frames.append(
                    frame
                )

        except Exception as error:
            print(
                f"  FAILED: {error}"
            )

            failed.append(
                {
                    "playerId": player_id,
                    "name": name,
                    "error": str(error)
                }
            )

        if index < len(players) - 1:
            delay = random.uniform(
                REQUEST_DELAY_MIN,
                REQUEST_DELAY_MAX
            )

            time.sleep(
                delay
            )

    if not frames:
        raise RuntimeError(
            "No MoneyPuck goalie game data downloaded."
        )

    goalies = pd.concat(
        frames,
        ignore_index=True
    )

    if (
        "gameId" in goalies.columns
        and "situation" in goalies.columns
    ):
        goalies = goalies.drop_duplicates(
            subset=[
                "playerId",
                "gameId",
                "situation"
            ]
        )

    goalies = goalies.sort_values(
        [
            "season",
            "gameId",
            "playerId"
        ],
        na_position="last"
    ).reset_index(
        drop=True
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
        f"Goalies downloaded: "
        f"{goalies['playerId'].nunique():,}"
    )

    print(
        f"Goalies failed: "
        f"{len(failed):,}"
    )

    print(
        f"Saved: "
        f"{OUTPUT_FILE}"
    )

    if failed:
        print()
        print("FAILED GOALIES")
        print("-" * 60)

        for goalie in failed:
            print(
                f"{goalie['name']} "
                f"({goalie['playerId']}): "
                f"{goalie['error']}"
            )

    print()
    print("AVAILABLE COLUMNS")
    print("-" * 60)

    for column in goalies.columns:
        print(column)


if __name__ == "__main__":
    main()