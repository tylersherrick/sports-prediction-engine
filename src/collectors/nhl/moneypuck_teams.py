from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import random
import time

import pandas as pd


OUTPUT_FILE = Path(
    "data/processed/nhl_moneypuck_team_games.csv"
)

SEASONS = [
    2022,
    2023,
    2024,
    2025,
    2026
]

TEAMS = [
    "ANA",
    "BOS",
    "BUF",
    "CAR",
    "CBJ",
    "CGY",
    "CHI",
    "COL",
    "DAL",
    "DET",
    "EDM",
    "FLA",
    "LAK",
    "MIN",
    "MTL",
    "NJD",
    "NSH",
    "NYI",
    "NYR",
    "OTT",
    "PHI",
    "PIT",
    "SEA",
    "SJS",
    "STL",
    "TBL",
    "TOR",
    "UTA",
    "VAN",
    "VGK",
    "WPG",
    "WSH"
]

BASE_URL = (
    "https://moneypuck.com/moneypuck/playerData/"
    "teamGameByGame/{season}/regular/{team}.csv"
)

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


def download_team(
    season,
    team
):
    url = BASE_URL.format(
        season=season,
        team=team
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
                    f"    Rate limited. "
                    f"Waiting {wait_time}s..."
                )

                time.sleep(
                    wait_time
                )

                continue

            if error.code == 404:
                return None

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
                    f"    Request failed. "
                    f"Retrying in {wait_time}s..."
                )

                time.sleep(
                    wait_time
                )

                continue

            raise


def main():
    print()
    print("MONEYPUCK NHL TEAM GAME COLLECTOR")
    print("=" * 60)

    frames = []
    failed = []

    total_requests = (
        len(SEASONS)
        * len(TEAMS)
    )

    request_number = 0

    for season in SEASONS:
        print()
        print(
            f"{season}-{season + 1}"
        )
        print("-" * 60)

        for team in TEAMS:
            request_number += 1

            print(
                f"[{request_number}/{total_requests}] "
                f"{team}"
            )

            try:
                frame = download_team(
                    season,
                    team
                )

                if frame is None:
                    print(
                        "  No data"
                    )

                elif frame.empty:
                    print(
                        "  Empty"
                    )

                else:
                    frame["season"] = season

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

                failed.append(
                    {
                        "season": season,
                        "team": team,
                        "error": str(error)
                    }
                )

            if request_number < total_requests:
                time.sleep(
                    random.uniform(
                        REQUEST_DELAY_MIN,
                        REQUEST_DELAY_MAX
                    )
                )

    if not frames:
        raise RuntimeError(
            "No MoneyPuck team game data downloaded."
        )

    teams = pd.concat(
        frames,
        ignore_index=True
    )

    duplicate_columns = [
        column
        for column in [
            "team",
            "gameId",
            "situation"
        ]
        if column in teams.columns
    ]

    if len(duplicate_columns) == 3:
        teams = teams.drop_duplicates(
            subset=duplicate_columns
        )

    sort_columns = [
        column
        for column in [
            "season",
            "gameId",
            "team"
        ]
        if column in teams.columns
    ]

    if sort_columns:
        teams = teams.sort_values(
            sort_columns,
            na_position="last"
        ).reset_index(
            drop=True
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    teams.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 60)

    print(
        f"Total rows: "
        f"{len(teams):,}"
    )

    print(
        f"Columns: "
        f"{len(teams.columns):,}"
    )

    if "team" in teams.columns:
        print(
            f"Teams: "
            f"{teams['team'].nunique():,}"
        )

    if "gameId" in teams.columns:
        print(
            f"Games: "
            f"{teams['gameId'].nunique():,}"
        )

    print(
        f"Failed downloads: "
        f"{len(failed):,}"
    )

    print(
        f"Saved: "
        f"{OUTPUT_FILE}"
    )

    if failed:
        print()
        print("FAILED DOWNLOADS")
        print("-" * 60)

        for item in failed:
            print(
                f"{item['season']} "
                f"{item['team']}: "
                f"{item['error']}"
            )

    print()
    print("AVAILABLE COLUMNS")
    print("-" * 60)

    for column in teams.columns:
        print(column)


if __name__ == "__main__":
    main()