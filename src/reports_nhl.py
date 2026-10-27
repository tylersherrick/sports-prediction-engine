from datetime import datetime
from zoneinfo import ZoneInfo


def nhl_skater_rows(skaters):
    return "".join(
        f"""
        <tr>
            <td>
                <div class="player-name">
                    {player["player"]}
                </div>
                <div class="player-meta">
                    {player["team"]} ·
                    {player.get("position") or "-"}
                </div>
            </td>
            <td>{player["shots_on_goal"]:.2f}</td>
            <td>{player["points"]:.2f}</td>
            <td>{player["goals"]:.2f}</td>
            <td>{player["assists"]:.2f}</td>
        </tr>
        """
        for player in skaters
    )


def nhl_goalie_rows(goalies):
    return "".join(
        f"""
        <tr>
            <td>
                <div class="player-name">
                    {goalie["player"]}
                </div>
                <div class="player-meta">
                    {goalie["team"]}
                </div>
            </td>
            <td>{goalie["saves"]:.2f}</td>
        </tr>
        """
        for goalie in goalies
    )


def build_nhl_report(result):
    winner = result["winner"]
    puck_line = result["puck_line"]
    total = result["total"]
    skaters = result.get("skaters", [])
    goalies = result.get("goalies", [])

    away_logo = result.get("away_logo", "")
    home_logo = result.get("home_logo", "")

    game_date = datetime.fromisoformat(
        result["game_date"]
    )
    game_date = game_date.astimezone(
        ZoneInfo("America/Chicago")
    )
    game_date_display = game_date.strftime(
        "%A, %B %d, %Y • %I:%M %p CT"
    ).replace(" 0", " ")

    skater_rows = nhl_skater_rows(skaters)
    goalie_rows = nhl_goalie_rows(goalies)

    market_odds = result.get("market_odds") or {}
    moneyline_value = (
        result.get("moneyline_value") or []
    )
    market_comparison = (
        result.get("market_comparison") or {}
    )

    puck_market = (
        market_comparison.get("puck_line") or {}
    )

    sportsbook = (
        market_odds.get("provider")
        or "Sportsbook"
    )

    market_home_line = puck_market.get(
        "market_home_line"
    )
    puck_lean = puck_market.get("lean")
    puck_margin_edge = puck_market.get(
        "margin_edge"
    )
    puck_home_odds = puck_market.get(
        "home_odds"
    )
    puck_away_odds = puck_market.get(
        "away_odds"
    )

    if (
        market_home_line is not None
        and puck_lean in ("home", "away")
    ):
        puck_lean_team = (
            result["home_team"]
            if puck_lean == "home"
            else result["away_team"]
        )

        puck_lean_line = (
            float(market_home_line)
            if puck_lean == "home"
            else -float(market_home_line)
        )

        puck_lean_odds = (
            puck_home_odds
            if puck_lean == "home"
            else puck_away_odds
        )

        puck_lean_odds_display = (
            f"{float(puck_lean_odds):+.0f}"
            if puck_lean_odds is not None
            else "Unavailable"
        )

        puck_market_display = (
            f"{puck_lean_team} "
            f"{puck_lean_line:+.1f} "
            f"({puck_lean_odds_display})"
        )
    else:
        puck_market_display = "Unavailable"

    puck_edge_display = (
        f"{abs(float(puck_margin_edge)):.2f} goals"
        if puck_margin_edge is not None
        else "Unavailable"
    )

    total_market = (
        market_comparison.get("total") or {}
    )
    total_market_line = total_market.get(
        "market_line"
    )
    total_lean = total_market.get("lean")
    total_goal_edge = total_market.get(
        "goal_edge"
    )
    total_over_odds = total_market.get(
        "over_odds"
    )
    total_under_odds = total_market.get(
        "under_odds"
    )

    if total_market_line is not None:
        total_market_display = (
            f"{float(total_market_line):.1f}"
        )
    else:
        total_market_display = "Unavailable"

    if total_lean in ("over", "under"):
        total_lean_odds = (
            total_over_odds
            if total_lean == "over"
            else total_under_odds
        )

        total_lean_odds_display = (
            f"{float(total_lean_odds):+.0f}"
            if total_lean_odds is not None
            else "Unavailable"
        )

        total_lean_display = (
            f"{total_lean.upper()} "
            f"{total_market_display} "
            f"({total_lean_odds_display})"
        )
    else:
        total_lean_display = "Unavailable"

    total_edge_display = (
        f"{abs(float(total_goal_edge)):.2f} goals"
        if total_goal_edge is not None
        else "Unavailable"
    )

    moneyline_rows = ""

    for item in moneyline_value:
        odds = item.get("market_odds")

        odds_display = (
            f"{float(odds):+.0f}"
            if odds is not None
            else "Unavailable"
        )

        ev = item.get("expected_value")

        ev_display = (
            f"{ev:+.1%}"
            if ev is not None
            else "Unavailable"
        )

        ev_class = (
            "positive"
            if ev is not None and ev > 0
            else "negative"
            if ev is not None and ev < 0
            else ""
        )

        moneyline_rows += f"""
            <div class="row">
                <span class="label">
                    {item.get("team", "Unknown")}
                    · {odds_display}
                </span>

                <span class="value {ev_class}">
                    EV {ev_display}
                </span>
            </div>
        """

    best_moneyline = next(
        (
            item
            for item in moneyline_value
            if item.get("expected_value", 0) > 0
        ),
        None,
    )

    if best_moneyline:
        best_odds = best_moneyline.get(
            "market_odds"
        )

        best_odds_display = (
            f"{float(best_odds):+.0f}"
            if best_odds is not None
            else "Unavailable"
        )

        best_model_probability = (
            f'{best_moneyline["model_probability"]:.1%}'
        )
        best_breakeven_probability = (
            f'{best_moneyline["breakeven_probability"]:.1%}'
        )
        best_edge = (
            f'{best_moneyline["edge"]:+.1%}'
        )
        best_expected_value = (
            f'{best_moneyline["expected_value"]:+.1%}'
        )

        best_value_html = f"""
            <div class="card mobile-collapsible">
                <div
                    class="section-header"
                    onclick="toggleMobileCard(this)"
                >
                    <h2>MONEYLINE VALUE</h2>
                    <span class="collapse-arrow">▾</span>
                </div>

                <div class="collapsible-content">
                    <div class="card-body">
                        <div class="pick positive">
                            {best_moneyline["team"]}
                            {best_odds_display}
                        </div>
                    </div>

                    <div class="row">
                        <span class="label">
                            Model Probability
                        </span>
                        <span class="value">
                            {best_model_probability}
                        </span>
                    </div>

                    <div class="row">
                        <span class="label">
                            Breakeven Probability
                        </span>
                        <span class="value">
                            {best_breakeven_probability}
                        </span>
                    </div>

                    <div class="row">
                        <span class="label">
                            Model Edge
                        </span>
                        <span class="value positive">
                            {best_edge}
                        </span>
                    </div>

                    <div class="row">
                        <span class="label">
                            Expected Value
                        </span>
                        <span class="value positive">
                            {best_expected_value}
                        </span>
                    </div>
                </div>
            </div>
        """
    elif moneyline_value:
        best_value_html = """
            <div class="card mobile-collapsible">
                <div
                    class="section-header"
                    onclick="toggleMobileCard(this)"
                >
                    <h2>MONEYLINE VALUE</h2>
                    <span class="collapse-arrow">▾</span>
                </div>

                <div class="collapsible-content">
                    <div class="card-body">
                        <div class="note">
                            No positive-value moneyline
                            identified.
                        </div>
                    </div>
                </div>
            </div>
        """
    else:
        best_value_html = """
            <div class="card mobile-collapsible">
                <div
                    class="section-header"
                    onclick="toggleMobileCard(this)"
                >
                    <h2>MONEYLINE VALUE</h2>
                    <span class="collapse-arrow">▾</span>
                </div>

                <div class="collapsible-content">
                    <div class="card-body">
                        <div class="note">
                            Sportsbook odds unavailable.
                        </div>
                    </div>
                </div>
            </div>
        """

    if not skater_rows:
        skater_rows = (
            '<tr><td colspan="5" class="empty">'
            "No skater projections available."
            "</td></tr>"
        )

    if not goalie_rows:
        goalie_rows = (
            '<tr><td colspan="2" class="empty">'
            "No goalie projections available."
            "</td></tr>"
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
            {result["away_team"]} @
            {result["home_team"]}
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
                margin: 0;
            }}

            .winner-pick {{
                font-size: 19px;
                font-weight: 750;
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

            .nhl-percent {{
                display: none;
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

            .positive {{
                color: #16813b;
            }}

            .negative {{
                color: #b42318;
            }}

            .note {{
                color: #666;
                font-size: 13px;
                line-height: 1.4;
            }}

            .table-wrap {{
                overflow-x: auto;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
            }}

            th {{
                background: #f8f8f8;
                color: #666;
                font-size: 11px;
                letter-spacing: .5px;
                text-align: right;
                padding: 10px 14px;
                border-bottom: 1px solid #ddd;
            }}

            th:first-child {{
                text-align: left;
            }}

            td {{
                padding: 11px 14px;
                text-align: right;
                font-weight: 600;
                border-top: 1px solid #e3e3e3;
                white-space: nowrap;
            }}

            td:first-child {{
                text-align: left;
                white-space: normal;
            }}

            .player-name {{
                font-weight: 700;
            }}

            .player-meta {{
                color: #888;
                font-size: 12px;
                font-weight: 400;
                margin-top: 2px;
            }}

            .goalie-table-wrap {{
                overflow: hidden;
                width: 100%;
            }}

            .goalie-table {{
                width: 100%;
                min-width: 0;
                table-layout: fixed;
            }}

            .goalie-table th:first-child,
            .goalie-table td:first-child {{
                width: 72%;
                text-align: left;
            }}

            .goalie-table th:last-child,
            .goalie-table td:last-child {{
                width: 28%;
                text-align: right;
            }}

            .empty {{
                color: #777;
                text-align: left;
                font-weight: 400;
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

                .skater-table {{
                    min-width: 560px;
                }}

                .goalie-table {{
                    min-width: 0;
                    width: 100%;
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
                            id="nhl-odds-button"
                            class="active"
                            onclick="event.stopPropagation(); showNhlOdds()"
                        >
                            Odds
                        </button>

                        <button
                            id="nhl-percent-button"
                            onclick="event.stopPropagation(); showNhlPercent()"
                        >
                            %
                        </button>
                    </div>

                    <span class="collapse-arrow">▾</span>
                </div>

                <div class="collapsible-content">
                    <div class="card-body">
                        <div class="winner-pick">
                            {winner["pick"]}
                        </div>
                    </div>

                    <div class="row">
                        <span class="label">
                            Model Fair Odds
                        </span>

                        <span class="value nhl-odds">
                            {winner["fair_odds"]:+d}
                        </span>

                        <span class="value nhl-percent">
                            {winner["probability"]:.1%}
                        </span>
                    </div>

                    <div class="row">
                        <span class="label">
                            {result["away_team"]}
                        </span>

                        <span class="value">
                            {winner["away_probability"]:.1%}
                        </span>
                    </div>

                    <div class="row">
                        <span class="label">
                            {result["home_team"]}
                        </span>

                        <span class="value">
                            {winner["home_probability"]:.1%}
                        </span>
                    </div>

                    <div class="section-header">
                        <h2>
                            SPORTSBOOK MONEYLINE ·
                            {sportsbook}
                        </h2>
                    </div>

                    {
                        moneyline_rows
                        if moneyline_rows
                        else (
                            '<div class="card-body">'
                            '<div class="note">'
                            'Sportsbook odds unavailable.'
                            '</div></div>'
                        )
                    }
                </div>
            </div>

            <div class="prediction-grid">
                <div class="card mobile-collapsible">
                    <div
                        class="section-header"
                        onclick="toggleMobileCard(this)"
                    >
                        <h2>PUCK LINE PREDICTION</h2>
                        <span class="collapse-arrow">▾</span>
                    </div>

                    <div class="collapsible-content">
                        <div class="card-body">
                            <div class="pick">
                                {puck_line["team"]}
                                {puck_line["line"]:+.2f}
                            </div>
                        </div>

                        <div class="row">
                            <span class="label">
                                Predicted Home Margin
                            </span>

                            <span class="value">
                                {puck_line["predicted_home_margin"]:+.2f}
                            </span>
                        </div>

                        <div class="row">
                            <span class="label">
                                Market / Model Lean
                            </span>

                            <span class="value">
                                {puck_market_display}
                            </span>
                        </div>

                        <div class="row">
                            <span class="label">
                                Model Edge
                            </span>

                            <span class="value">
                                {puck_edge_display}
                            </span>
                        </div>
                    </div>
                </div>

                <div class="card mobile-collapsible">
                    <div
                        class="section-header"
                        onclick="toggleMobileCard(this)"
                    >
                        <h2>TOTAL PREDICTION</h2>
                        <span class="collapse-arrow">▾</span>
                    </div>

                    <div class="collapsible-content">
                        <div class="card-body">
                            <div class="pick">
                                {total["predicted_total"]:.2f}
                            </div>
                        </div>

                        <div class="row">
                            <span class="label">
                                Market Total
                            </span>

                            <span class="value">
                                {total_market_display}
                            </span>
                        </div>

                        <div class="row">
                            <span class="label">
                                Model Lean
                            </span>

                            <span class="value">
                                {total_lean_display}
                            </span>
                        </div>

                        <div class="row">
                            <span class="label">
                                Model Edge
                            </span>

                            <span class="value">
                                {total_edge_display}
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {best_value_html}

            <div class="card mobile-collapsible">
                <div
                    class="section-header"
                    onclick="toggleMobileCard(this)"
                >
                    <h2>SKATER PROJECTIONS</h2>
                    <span class="collapse-arrow">▾</span>
                </div>

                <div class="collapsible-content">
                    <div class="table-wrap">
                        <table class="skater-table">
                            <thead>
                                <tr>
                                    <th>Player</th>
                                    <th>SOG</th>
                                    <th>PTS</th>
                                    <th>G</th>
                                    <th>A</th>
                                </tr>
                            </thead>

                            <tbody>
                                {skater_rows}
                            </tbody>
                        </table>
                    </div>

                    <div class="card-body">
                        <div class="note">
                            Model stat projections.
                            Sportsbook player-prop
                            comparisons will be added
                            separately.
                        </div>
                    </div>
                </div>
            </div>

            <div class="card mobile-collapsible">
                <div
                    class="section-header"
                    onclick="toggleMobileCard(this)"
                >
                    <h2>GOALIE SAVES</h2>
                    <span class="collapse-arrow">▾</span>
                </div>

                <div class="collapsible-content">
                    <div class="goalie-table-wrap">
                        <table class="goalie-table">
                            <thead>
                                <tr>
                                    <th>Goalie</th>
                                    <th>Saves</th>
                                </tr>
                            </thead>

                            <tbody>
                                {goalie_rows}
                            </tbody>
                        </table>
                    </div>

                    <div class="card-body">
                        <div class="note">
                            Multiple goalies may appear
                            until the expected starter
                            is identified.
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            function showNhlOdds() {{
                document
                    .querySelectorAll(".nhl-odds")
                    .forEach(element => {{
                        element.style.display = "inline";
                    }});

                document
                    .querySelectorAll(".nhl-percent")
                    .forEach(element => {{
                        element.style.display = "none";
                    }});

                document
                    .getElementById(
                        "nhl-odds-button"
                    )
                    .classList.add("active");

                document
                    .getElementById(
                        "nhl-percent-button"
                    )
                    .classList.remove("active");
            }}

            function showNhlPercent() {{
                document
                    .querySelectorAll(".nhl-odds")
                    .forEach(element => {{
                        element.style.display = "none";
                    }});

                document
                    .querySelectorAll(".nhl-percent")
                    .forEach(element => {{
                        element.style.display = "inline";
                    }});

                document
                    .getElementById(
                        "nhl-percent-button"
                    )
                    .classList.add("active");

                document
                    .getElementById(
                        "nhl-odds-button"
                    )
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