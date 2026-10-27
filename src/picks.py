from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from src.services.schedule import get_nfl_games, get_nhl_games


CENTRAL = ZoneInfo("America/Chicago")


def get_team(game, home_away):
    competition = game["competitions"][0]

    return next(
        competitor
        for competitor in competition["competitors"]
        if competitor["homeAway"] == home_away
    )


def build_game_card(game, league):
    away = get_team(game, "away")
    home = get_team(game, "home")

    away_team = away["team"]
    home_team = home["team"]

    away_logo = away_team.get("logo", "")
    home_logo = home_team.get("logo", "")

    game_time = datetime.fromisoformat(
        game["date"].replace("Z", "+00:00")
    ).astimezone(CENTRAL)

    time_display = game_time.strftime(
        "%I:%M %p CT"
    ).lstrip("0")

    game_id = game["id"]

    return f"""
        <a
            class="game-card"
            data-league="{league}"
            href="/{league}/{game_id}/report"
        >
            <div class="matchup">
                <div class="team">
                    <img
                        src="{away_logo}"
                        alt="{away_team["displayName"]}"
                    >
                    <span>
                        {away_team["abbreviation"]}
                    </span>
                </div>

                <div class="versus">
                    VS
                </div>

                <div class="team">
                    <img
                        src="{home_logo}"
                        alt="{home_team["displayName"]}"
                    >
                    <span>
                        {home_team["abbreviation"]}
                    </span>
                </div>
            </div>

            <div class="game-time">
                {time_display}
            </div>
        </a>
    """


def build_league_section(title, games, league):
    if not games:
        return ""

    cards = "".join(
        build_game_card(game, league)
        for game in games
    )

    return f"""
        <section
            class="league-section"
            data-league="{league}"
        >
            <h2>{title}</h2>

            <div class="games">
                {cards}
            </div>
        </section>
    """


def build_league_filters(nfl_games, nhl_games):
    leagues = []

    if nhl_games:
        leagues.append(("nhl", "NHL"))

    if nfl_games:
        leagues.append(("nfl", "NFL"))

    if len(leagues) <= 1:
        return ""

    buttons = """
        <button
            class="filter-button active"
            onclick="filterLeague('all', this)"
        >
            All
        </button>
    """

    for league, label in leagues:
        buttons += f"""
            <button
                class="filter-button"
                onclick="filterLeague('{league}', this)"
            >
                {label}
            </button>
        """

    return f"""
        <div class="filters">
            {buttons}
        </div>
    """


def build_date_picker(selected_date):
    previous_date = selected_date - timedelta(days=1)
    next_date = selected_date + timedelta(days=1)

    return f"""
        <div class="date-picker">
            <a
                class="date-arrow"
                href="/picks?date={previous_date.isoformat()}"
            >
                ←
            </a>

            <input
                type="date"
                value="{selected_date.isoformat()}"
                onchange="changeDate(this.value)"
            >

            <a
                class="date-arrow"
                href="/picks?date={next_date.isoformat()}"
            >
                →
            </a>
        </div>
    """


def build_picks_page(selected_date=None):
    if selected_date is None:
        selected_date = date.today()

    nfl_games = get_nfl_games(selected_date)
    nhl_games = get_nhl_games(selected_date)

    date_picker = build_date_picker(
        selected_date
    )

    filters = build_league_filters(
        nfl_games,
        nhl_games,
    )

    nfl_section = build_league_section(
        "NFL",
        nfl_games,
        "nfl",
    )

    nhl_section = build_league_section(
        "NHL",
        nhl_games,
        "nhl",
    )

    if not nfl_section and not nhl_section:
        content = f"""
            <div class="empty">
                No NFL or NHL games on
                {selected_date.strftime("%B %d, %Y")}.
            </div>
        """
    else:
        content = nhl_section + nfl_section

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta
            name="viewport"
            content="width=device-width, initial-scale=1"
        >

        <title>Picks</title>

        <style>
            * {{
                box-sizing: border-box;
            }}

            body {{
                font-family:
                    -apple-system,
                    BlinkMacSystemFont,
                    "Segoe UI",
                    Roboto,
                    Helvetica,
                    Arial,
                    sans-serif;
                background: #f5f5f5;
                color: #171717;
                margin: 0;
                padding: 32px 20px 60px;
                border-top: 10px solid #333;
            }}

            .container {{
                max-width: 900px;
                margin: auto;
            }}

            h1 {{
                margin: 0 0 18px;
                font-size: 28px;
                font-weight: 750;
            }}

            h2 {{
                color: #666;
                font-size: 12px;
                font-weight: 750;
                letter-spacing: .5px;
                margin: 0 0 12px;
            }}

            .date-picker {{
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 18px;
            }}

            .date-picker input {{
                height: 38px;
                background: #fff;
                color: #171717;
                border: 1px solid #d6d6d6;
                border-radius: 7px;
                padding: 8px 12px;
                font-family: inherit;
                font-size: 14px;
            }}

            .date-arrow {{
                display: flex;
                align-items: center;
                justify-content: center;
                width: 38px;
                height: 38px;
                background: #fff;
                color: #333;
                border: 1px solid #d6d6d6;
                border-radius: 7px;
                text-decoration: none;
                font-size: 18px;
                font-weight: 700;
            }}

            .date-arrow:hover {{
                background: #f1f1f1;
            }}

            .filters {{
                display: flex;
                gap: 8px;
                margin-bottom: 30px;
            }}

            .filter-button {{
                border: 1px solid #d6d6d6;
                background: #fff;
                color: #666;
                padding: 8px 14px;
                border-radius: 7px;
                cursor: pointer;
                font-family: inherit;
                font-size: 13px;
                font-weight: 700;
            }}

            .filter-button:hover {{
                background: #f1f1f1;
            }}

            .filter-button.active {{
                background: #333;
                border-color: #333;
                color: #fff;
            }}

            .league-section {{
                margin-bottom: 35px;
            }}

            .games {{
                display: grid;
                grid-template-columns:
                    repeat(auto-fill, minmax(220px, 1fr));
                gap: 12px;
            }}

            .game-card {{
                display: block;
                background: #fff;
                border: 1px solid #d6d6d6;
                border-radius: 10px;
                padding: 18px;
                color: #171717;
                text-decoration: none;
                transition:
                    transform .15s ease,
                    background .15s ease,
                    border-color .15s ease;
            }}

            .game-card:hover {{
                background: #fafafa;
                border-color: #bcbcbc;
                transform: translateY(-2px);
            }}

            .matchup {{
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 18px;
            }}

            .team {{
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 8px;
                min-width: 58px;
                font-size: 14px;
                font-weight: 750;
            }}

            .team img {{
                width: 52px;
                height: 52px;
                object-fit: contain;
            }}

            .versus {{
                color: #999;
                font-size: 10px;
                font-weight: 800;
                letter-spacing: .5px;
            }}

            .game-time {{
                text-align: center;
                color: #666;
                font-size: 13px;
                margin-top: 14px;
            }}

            .empty {{
                color: #666;
                background: #fff;
                border: 1px solid #d6d6d6;
                border-radius: 8px;
                padding: 20px;
                font-size: 14px;
            }}

            @media (max-width: 600px) {{
                body {{
                    padding: 18px 12px 40px;
                }}

                h1 {{
                    font-size: 24px;
                }}

                .filters {{
                    margin-bottom: 24px;
                }}

                .games {{
                    grid-template-columns: 1fr;
                }}

                .game-card {{
                    padding: 16px;
                }}

                .team img {{
                    width: 56px;
                    height: 56px;
                }}
            }}
        </style>
    </head>

    <body>
        <div class="container">
            <h1>Picks</h1>

            {date_picker}

            {filters}

            {content}
        </div>

        <script>
            function changeDate(value) {{
                window.location.href =
                    "/picks?date=" + value;
            }}

            function filterLeague(league, button) {{
                document
                    .querySelectorAll(".filter-button")
                    .forEach(element => {{
                        element.classList.remove("active");
                    }});

                button.classList.add("active");

                document
                    .querySelectorAll(".league-section")
                    .forEach(section => {{
                        if (
                            league === "all" ||
                            section.dataset.league === league
                        ) {{
                            section.style.display = "block";
                        }} else {{
                            section.style.display = "none";
                        }}
                    }});
            }}
        </script>
    </body>
    </html>
    """