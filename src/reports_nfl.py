from datetime import datetime
from zoneinfo import ZoneInfo


def build_nfl_report(result):
    winner = result["winner"]
    spread = result["spread"]
    total = result["total"]

    game_date = datetime.fromisoformat(result["game_date"])
    game_date = game_date.astimezone(
        ZoneInfo("America/Chicago")
    )
    game_date_display = game_date.strftime(
        "%A, %B %d, %Y • %I:%M %p CT"
    ).replace(" 0", " ")

    winner_market_odds = (
        f'{winner["market_odds"]:+d}'
        if winner["market_odds"] is not None
        else "Unavailable"
    )

    winner_breakeven = (
        f'{winner["market_breakeven_probability"]:.1%}'
        if winner["market_breakeven_probability"] is not None
        else "Unavailable"
    )

    value_pick = (
        winner["value_pick"]
        if winner["value_pick"] is not None
        else "No positive value"
    )

    value_market_odds = (
        f'{winner["value_market_odds"]:+d}'
        if winner["value_market_odds"] is not None
        else "Unavailable"
    )

    value_model_probability = (
        f'{winner["value_model_probability"]:.1%}'
        if winner["value_model_probability"] is not None
        else "Unavailable"
    )

    value_breakeven = (
        f'{winner["value_breakeven_probability"]:.1%}'
        if winner["value_breakeven_probability"] is not None
        else "Unavailable"
    )

    value_edge = (
        f'+{winner["value_edge"]:.1%}'
        if winner["value_edge"] is not None
        else "Unavailable"
    )

    expected_value = (
        f'+{winner["expected_value"]:.1%}'
        if winner["expected_value"] is not None
        else "Unavailable"
    )

    spread_market = (
        f'{spread["market_team"]} '
        f'{spread["market_line"]:+.1f}'
        if spread["market_line"] is not None
        else "Unavailable"
    )

    spread_edge = (
        f'{spread["edge_team"]} '
        f'{spread["edge"]:.1f} points'
        if spread["edge"] is not None
        else "Unavailable"
    )

    total_market = (
        f'{total["market_total"]:.1f}'
        if total["market_total"] is not None
        else "Unavailable"
    )

    total_lean = (
        total["lean"].title()
        if total["lean"] is not None
        else "Unavailable"
    )

    total_edge = (
        f'{total["edge"]:.1f} points'
        if total["edge"] is not None
        else "Unavailable"
    )

    away_logo = (
        "https://a.espncdn.com/i/teamlogos/nfl/500/"
        f'{result["away_abbreviation"].lower()}.png'
    )

    home_logo = (
        "https://a.espncdn.com/i/teamlogos/nfl/500/"
        f'{result["home_abbreviation"].lower()}.png'
    )

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta
            name="viewport"
            content="width=device-width, initial-scale=1"
        >

        <title>
            {result["away_team"]} @ {result["home_team"]}
        </title>

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

            .back-button {{
                display: inline-flex;
                align-items: center;
                min-height: 38px;
                margin-bottom: 14px;
                padding: 8px 12px;
                background: #fff;
                color: #333;
                border: 1px solid #d6d6d6;
                border-radius: 7px;
                text-decoration: none;
                font-size: 13px;
                font-weight: 700;
            }}

            .back-button:hover {{
                background: #f1f1f1;
            }}

            .matchup-card {{
                display: grid;
                grid-template-columns: 1fr auto 1fr;
                align-items: center;
                gap: 30px;
                background: #fff;
                border: 1px solid #d6d6d6;
                border-radius: 10px;
                padding: 24px 30px;
                margin-bottom: 20px;
            }}

            .team {{
                display: flex;
                align-items: center;
                gap: 16px;
            }}

            .team.home {{
                flex-direction: row-reverse;
                text-align: right;
            }}

            .team-logo {{
                width: 76px;
                height: 76px;
                object-fit: contain;
            }}

            .team-name {{
                font-size: 20px;
                font-weight: 750;
                line-height: 1.15;
            }}

            .team-label {{
                color: #888;
                font-size: 12px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: .5px;
                margin-top: 5px;
            }}

            .at {{
                color: #999;
                font-size: 12px;
                font-weight: 800;
                letter-spacing: 1px;
            }}

            .game-info {{
                margin-bottom: 24px;
            }}

            .game-date {{
                color: #666;
                font-size: 14px;
                margin-bottom: 3px;
            }}

            .game-id {{
                color: #888;
                font-size: 12px;
            }}

            .card {{
                background: #fff;
                border: 1px solid #d6d6d6;
                border-radius: 8px;
                margin-bottom: 20px;
                overflow: hidden;
            }}

            .card-header,
            .section-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                min-height: 40px;
                padding: 10px 14px;
                background: #f1f1f1;
                border-bottom: 1px solid #d6d6d6;
            }}

            h2 {{
                font-size: 12px;
                color: #666;
                margin: 0;
                letter-spacing: .5px;
                font-weight: 750;
            }}

            .card-body {{
                padding: 16px 14px;
            }}

            .pick {{
                font-size: 18px;
                font-weight: 750;
                margin-bottom: 8px;
            }}

            .winner-pick {{
                font-size: 19px;
                font-weight: 750;
            }}

            .winner-probability {{
                font-size: 34px;
                font-weight: 750;
                line-height: 1;
                margin: 10px 0 4px;
            }}

            .probability-label {{
                color: #666;
                font-size: 13px;
                margin-bottom: 8px;
            }}

            .row {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 20px;
                padding: 12px 14px;
                border-top: 1px solid #e3e3e3;
                font-size: 14px;
            }}

            .label {{
                color: #666;
            }}

            .value {{
                text-align: right;
                font-weight: 700;
                color: #222;
            }}

            .toggle {{
                display: flex;
                border: 1px solid #ccc;
                border-radius: 5px;
                overflow: hidden;
            }}

            .toggle button {{
                border: 0;
                background: #fff;
                color: #666;
                padding: 5px 10px;
                cursor: pointer;
                font-family: inherit;
                font-weight: 700;
            }}

            .toggle button.active {{
                background: #333;
                color: #fff;
            }}

            .prediction-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 14px;
                margin-bottom: 20px;
            }}

            .prediction-grid .card {{
                margin-bottom: 0;
            }}

            .explanation {{
                background: #f3f3f3;
                color: #555;
                border-top: 1px solid #d6d6d6;
                padding: 13px 14px;
                font-size: 13px;
                line-height: 1.4;
            }}

            .percent {{
                display: none;
            }}

            .collapse-arrow {{
                display: none;
                color: #666;
                font-size: 16px;
                line-height: 1;
                transition: transform .2s ease;
            }}

            @media (max-width: 650px) {{
                body {{
                    padding: 18px 12px 40px;
                }}

                .back-button {{
                    margin-bottom: 12px;
                }}

                .matchup-card {{
                    gap: 10px;
                    padding: 18px 12px;
                }}

                .team {{
                    flex-direction: column;
                    gap: 8px;
                    text-align: center;
                }}

                .team.home {{
                    flex-direction: column;
                    text-align: center;
                }}

                .team-logo {{
                    width: 60px;
                    height: 60px;
                }}

                .team-name {{
                    font-size: 15px;
                }}

                .team-label {{
                    font-size: 10px;
                }}

                .game-info {{
                    text-align: center;
                    margin-bottom: 20px;
                }}

                .prediction-grid {{
                    display: block;
                    margin-bottom: 0;
                }}

                .prediction-grid .card {{
                    margin-bottom: 12px;
                }}

                .card {{
                    margin-bottom: 12px;
                }}

                .row {{
                    gap: 12px;
                }}

                .mobile-collapsible
                .section-header,
                .mobile-collapsible
                .card-header {{
                    cursor: pointer;
                    user-select: none;
                }}

                .collapse-arrow {{
                    display: block;
                }}

                .mobile-collapsible.collapsed
                .collapsible-content {{
                    display: none;
                }}

                .mobile-collapsible.collapsed
                .section-header,
                .mobile-collapsible.collapsed
                .card-header {{
                    border-bottom: 0;
                }}

                .mobile-collapsible:not(.collapsed)
                .collapse-arrow {{
                    transform: rotate(180deg);
                }}

                .mobile-collapsible
                .toggle {{
                    margin-left: auto;
                    margin-right: 10px;
                }}
            }}
        </style>
    </head>

    <body>
        <div class="container">
            <a
                class="back-button"
                href="/picks?date={game_date.date().isoformat()}"
            >
                ← Back to Picks
            </a>

            <div class="matchup-card">
                <div class="team">
                    <img
                        class="team-logo"
                        src="{away_logo}"
                        alt="{result["away_team"]}"
                    >

                    <div>
                        <div class="team-name">
                            {result["away_team"]}
                        </div>

                        <div class="team-label">
                            Away
                        </div>
                    </div>
                </div>

                <div class="at">
                    AT
                </div>

                <div class="team home">
                    <img
                        class="team-logo"
                        src="{home_logo}"
                        alt="{result["home_team"]}"
                    >

                    <div>
                        <div class="team-name">
                            {result["home_team"]}
                        </div>

                        <div class="team-label">
                            Home
                        </div>
                    </div>
                </div>
            </div>

            <div class="game-info">
                <div class="game-date">
                    {game_date_display}
                </div>

                <div class="game-id">
                    ESPN Game ID: {result["game_id"]}
                </div>
            </div>

            <div class="card mobile-collapsible">
                <div
                    class="card-header"
                    onclick="toggleMobileCard(this)"
                >
                    <h2>WINNER PREDICTION</h2>

                    <div class="toggle">
                        <button
                            id="odds-button"
                            class="active"
                            onclick="event.stopPropagation(); showOdds()"
                        >
                            Odds
                        </button>

                        <button
                            id="percent-button"
                            onclick="event.stopPropagation(); showPercent()"
                        >
                            %
                        </button>
                    </div>

                    <span class="collapse-arrow">
                        ▾
                    </span>
                </div>

                <div class="collapsible-content">
                    <div class="card-body">
                        <div class="winner-pick">
                            {winner["pick"]}
                        </div>

                        <div class="winner-probability percent">
                            {winner["probability"]:.1%}
                        </div>

                        <div class="probability-label percent">
                            Win Probability
                        </div>
                    </div>

                    <div class="row">
                        <span class="label">
                            Model Fair Odds
                        </span>

                        <span class="value odds">
                            {winner["fair_odds"]:+d}
                        </span>

                        <span class="value percent">
                            {winner["probability"]:.1%}
                        </span>
                    </div>

                    <div class="row">
                        <span class="label">
                            Market Moneyline
                        </span>

                        <span class="value odds">
                            {winner_market_odds}
                        </span>

                        <span class="value percent">
                            {winner_breakeven}
                        </span>
                    </div>
                </div>
            </div>

            <div class="prediction-grid">
                <div class="card mobile-collapsible">
                    <div
                        class="section-header"
                        onclick="toggleMobileCard(this)"
                    >
                        <h2>SPREAD PREDICTION</h2>

                        <span class="collapse-arrow">
                            ▾
                        </span>
                    </div>

                    <div class="collapsible-content">
                        <div class="row">
                            <span class="label">
                                Model Prediction
                            </span>

                            <span class="value">
                                {spread["model_team"]}
                                {spread["model_line"]:+.1f}
                            </span>
                        </div>

                        <div class="row">
                            <span class="label">
                                Market Spread
                            </span>

                            <span class="value">
                                {spread_market}
                            </span>
                        </div>

                        <div class="row">
                            <span class="label">
                                Model Edge
                            </span>

                            <span class="value">
                                {spread_edge}
                            </span>
                        </div>

                        <div class="explanation">
                            Model sees {spread_edge}
                            versus the current market.
                        </div>
                    </div>
                </div>

                <div class="card mobile-collapsible">
                    <div
                        class="section-header"
                        onclick="toggleMobileCard(this)"
                    >
                        <h2>TOTAL PREDICTION</h2>

                        <span class="collapse-arrow">
                            ▾
                        </span>
                    </div>

                    <div class="collapsible-content">
                        <div class="row">
                            <span class="label">
                                Model Prediction
                            </span>

                            <span class="value">
                                {total_lean.upper()}
                                {total["model_total"]:.1f}
                            </span>
                        </div>

                        <div class="row">
                            <span class="label">
                                Market Total
                            </span>

                            <span class="value">
                                {total_market}
                            </span>
                        </div>

                        <div class="row">
                            <span class="label">
                                Model Edge
                            </span>

                            <span class="value">
                                {total_edge}
                            </span>
                        </div>

                        <div class="explanation">
                            Model projects {total_edge}
                            from the current market total.
                        </div>
                    </div>
                </div>
            </div>

            <div class="card mobile-collapsible">
                <div
                    class="section-header"
                    onclick="toggleMobileCard(this)"
                >
                    <h2>MONEYLINE VALUE</h2>

                    <span class="collapse-arrow">
                        ▾
                    </span>
                </div>

                <div class="collapsible-content">
                    <div class="card-body">
                        <div class="pick">
                            {value_pick} {value_market_odds}
                        </div>
                    </div>

                    <div class="row">
                        <span class="label">
                            Model Probability
                        </span>

                        <span class="value">
                            {value_model_probability}
                        </span>
                    </div>

                    <div class="row">
                        <span class="label">
                            Model Fair Odds
                        </span>

                        <span class="value">
                            {winner["fair_odds"]:+d}
                        </span>
                    </div>

                    <div class="row">
                        <span class="label">
                            Market Moneyline
                        </span>

                        <span class="value">
                            {value_market_odds}
                        </span>
                    </div>

                    <div class="row">
                        <span class="label">
                            Breakeven Probability
                        </span>

                        <span class="value">
                            {value_breakeven}
                        </span>
                    </div>

                    <div class="row">
                        <span class="label">
                            Model Edge
                        </span>

                        <span class="value">
                            {value_edge}
                        </span>
                    </div>

                    <div class="row">
                        <span class="label">
                            Expected Value
                        </span>

                        <span class="value">
                            {expected_value}
                        </span>
                    </div>
                </div>
            </div>
        </div>

        <script>
            function showOdds() {{
                document
                    .querySelectorAll(".odds")
                    .forEach(element => {{
                        element.style.display = "inline";
                    }});

                document
                    .querySelectorAll(".percent")
                    .forEach(element => {{
                        element.style.display = "none";
                    }});

                document
                    .getElementById("odds-button")
                    .classList.add("active");

                document
                    .getElementById("percent-button")
                    .classList.remove("active");
            }}

            function showPercent() {{
                document
                    .querySelectorAll(".odds")
                    .forEach(element => {{
                        element.style.display = "none";
                    }});

                document
                    .querySelectorAll(".percent")
                    .forEach(element => {{
                        element.style.display = "inline";
                    }});

                document
                    .getElementById("percent-button")
                    .classList.add("active");

                document
                    .getElementById("odds-button")
                    .classList.remove("active");
            }}

            function toggleMobileCard(header) {{
                if (
                    window.matchMedia(
                        "(max-width: 650px)"
                    ).matches
                ) {{
                    header
                        .closest(".mobile-collapsible")
                        .classList.toggle("collapsed");
                }}
            }}

            function setMobileCollapseState() {{
                const isMobile = window.matchMedia(
                    "(max-width: 650px)"
                ).matches;

                document
                    .querySelectorAll(
                        ".mobile-collapsible"
                    )
                    .forEach(card => {{
                        if (isMobile) {{
                            card.classList.add(
                                "collapsed"
                            );
                        }} else {{
                            card.classList.remove(
                                "collapsed"
                            );
                        }}
                    }});
            }}

            setMobileCollapseState();

            window.addEventListener(
                "resize",
                setMobileCollapseState
            );
        </script>
    </body>
    </html>
    """