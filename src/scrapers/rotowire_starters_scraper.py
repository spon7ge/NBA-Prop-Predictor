"""RotoWire daily starting lineups — NBA and WNBA.

Updates ``src/utils/team_info.py``:

* NBA  → ``projectedStartingFive`` / ``questionablePlayers``
* WNBA → ``projectedStartingFiveWnba`` / ``questionablePlayersWnba``

Examples::

    from src.scrapers.scrap_starters import NBADailyLineups, WNBADailyLineups

    nba = NBADailyLineups()
    nba.getDict()
    nba.updateTeamInfo()

    wnba = WNBADailyLineups()
    wnba.getDict()
    wnba.updateTeamInfo()
"""

from __future__ import annotations

import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

NBA_LINEUPS_URL = "https://www.rotowire.com/basketball/nba-lineups.php"
WNBA_LINEUPS_URL = "https://www.rotowire.com/wnba/lineups.php"

# Nickname (first token of RotoWire matchup label) → NBA abbreviation
NBA_TEAM_ABBREVIATIONS = {
    "Hawks": "ATL",
    "Celtics": "BOS",
    "Nets": "BKN",
    "Hornets": "CHA",
    "Bulls": "CHI",
    "Cavaliers": "CLE",
    "Mavericks": "DAL",
    "Nuggets": "DEN",
    "Pistons": "DET",
    "Warriors": "GSW",
    "Rockets": "HOU",
    "Pacers": "IND",
    "Clippers": "LAC",
    "Lakers": "LAL",
    "Grizzlies": "MEM",
    "Heat": "MIA",
    "Bucks": "MIL",
    "Timberwolves": "MIN",
    "Pelicans": "NOP",
    "Knicks": "NYK",
    "Thunder": "OKC",
    "Magic": "ORL",
    "76ers": "PHI",
    "Suns": "PHX",
    "Trail": "POR",
    "Kings": "SAC",
    "Spurs": "SAS",
    "Raptors": "TOR",
    "Jazz": "UTA",
    "Wizards": "WAS",
}

# Nickname → WNBA abbreviation (fallback if ``lineup__abbr`` missing)
WNBA_TEAM_ABBREVIATIONS = {
    "Dream": "ATL",
    "Sky": "CHI",
    "Sun": "CON",
    "Wings": "DAL",
    "Fever": "IND",
    "Sparks": "LAS",
    "Aces": "LVA",
    "Lynx": "MIN",
    "Liberty": "NYL",
    "Mercury": "PHO",
    "Storm": "SEA",
    "Mystics": "WAS",
    "Valkyries": "GSV",
    "Fire": "POR",
}


class DailyLineups:
    """Scrape RotoWire expected starting fives for ``nba`` or ``wnba``."""

    def __init__(self, league: str = "nba", url: str | None = None):
        league = league.strip().lower()
        if league not in ("nba", "wnba"):
            raise ValueError(f"league must be 'nba' or 'wnba', got {league!r}")
        self.league = league
        self.url = url or (NBA_LINEUPS_URL if league == "nba" else WNBA_LINEUPS_URL)
        self.team_abbreviations = (
            NBA_TEAM_ABBREVIATIONS if league == "nba" else WNBA_TEAM_ABBREVIATIONS
        )
        self.projected_dict_name = (
            "projectedStartingFive" if league == "nba" else "projectedStartingFiveWnba"
        )
        self.questionable_dict_name = (
            "questionablePlayers" if league == "nba" else "questionablePlayersWnba"
        )
        self.data: list[dict] = []
        self.soup = self._get_soup()

    def __str__(self) -> str:
        result = ""
        for index, matchup in enumerate(self.data):
            result += f"\n\nMatchup {index + 1}\n"
            for side in matchup:
                team = matchup[side]["team"]
                abbr = matchup[side].get("abbr") or ""
                label = f"{team} ({abbr})" if abbr else team
                result += f"\n\n{side} team: {label}\n{'-' * (len(label) + 12)}\n"
                result += "\nConfirmed Playing\n-----------------\n"
                for player in matchup[side]["confirmed"]:
                    result += player + "\n"
                result += "\nGame Time Decision\n------------------\n"
                for player in matchup[side]["gtd"]:
                    result += player + "\n"
                result += "\nQuestionable (May Not Play)\n---------------------------\n"
                for player in matchup[side]["questionable"]:
                    result += player + "\n"
                result += "\nConfirmed Out\n-------------\n"
                for player in matchup[side]["out"]:
                    result += player + "\n"
        return result

    def _get_soup(self) -> BeautifulSoup:
        response = requests.get(self.url, verify=True, timeout=30)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")

    def _matchup_cards(self):
        if self.league == "nba":
            cards = self.soup.find_all(
                "div",
                class_=lambda c: isinstance(c, list) and "lineup" in c and "is-nba" in c,
            )
            if cards:
                return cards
        # WNBA (and NBA fallback): top-level ``div.lineup`` game cards
        return [
            d
            for d in self.soup.find_all("div", class_="lineup")
            if "lineup__" not in " ".join(d.get("class") or [])
        ]

    @staticmethod
    def _side_abbr(matchup, side: str) -> str | None:
        """Read ``POR`` / ``NYL`` from ``lineup__abbr`` under visit/home team."""
        team_cls = "is-visit" if side == "away" else "is-home"
        team_el = matchup.select_one(f"a.lineup__team.{team_cls}")
        if team_el is None:
            return None
        abbr_el = team_el.select_one(".lineup__abbr")
        if abbr_el is None:
            return None
        text = abbr_el.get_text(strip=True)
        return text.upper() if text else None

    @staticmethod
    def _side_nickname(matchup, side: str) -> str:
        cls = (
            "lineup__mteam is-visit white"
            if side == "away"
            else "lineup__mteam is-home white"
        )
        el = matchup.find("a", {"class": cls})
        if el is None:
            # Newer markup sometimes omits ``white``
            sel = "a.lineup__mteam.is-visit" if side == "away" else "a.lineup__mteam.is-home"
            el = matchup.select_one(sel)
        if el is None:
            return ""
        return el.get_text(strip=True).split(None, 1)[0]

    def _resolve_abbr(self, matchup, side: str, nickname: str) -> str | None:
        abbr = self._side_abbr(matchup, side)
        if abbr:
            return abbr
        return self.team_abbreviations.get(nickname)

    def _get_injured_players(self, lineup_list) -> dict[str, set]:
        questionable: set[str] = set()
        out: set[str] = set()
        if lineup_list is None:
            return {"questionable": questionable, "out": out}

        injured_items = lineup_list.find_all(
            "li", class_=lambda x: x and "has-injury-status" in x
        )
        for item in injured_items:
            player_elem = item.find("a")
            if not player_elem:
                continue
            player_name = player_elem.get("title") or player_elem.get_text(strip=True)
            status_span = item.find("span", class_="lineup__inj")
            if status_span:
                status_text = status_span.get_text(strip=True).lower()
                if status_text == "out":
                    out.add(player_name)
                elif status_text in ("ques", "gtd", "questionable", "doubtful"):
                    questionable.add(player_name)
            else:
                title = (item.get("title") or "").lower()
                if "unlikely" in title:
                    out.add(player_name)
        return {"questionable": questionable, "out": out}

    @staticmethod
    def _ordered_expected_starters(lineup_list) -> list[dict[str, object]]:
        if lineup_list is None:
            return []
        allowed_titles = {"Very Likely To Play", "Likely To Play", "Toss Up To Play"}
        gtd_titles = {"Likely To Play", "Toss Up To Play"}
        out: list[dict[str, object]] = []
        for item in lineup_list.find_all("li", recursive=False):
            classes = item.get("class") or []
            if "lineup__title" in classes:
                break
            if "lineup__player" not in classes:
                continue
            title = item.get("title") or ""
            if title not in allowed_titles:
                continue
            link = item.find("a")
            if not link:
                continue
            name = (link.get("title") or link.get_text(strip=True) or "").strip()
            if not name:
                continue
            pos_el = item.find(class_="lineup__pos")
            position = pos_el.get_text(strip=True) if pos_el else None
            inj_el = item.find(class_="lineup__inj")
            inj_text = (inj_el.get_text(strip=True) if inj_el else "").casefold()
            # RotoWire marks GTD / questionable expected starters with injury
            # status or a non-"Very Likely" play-probability title.
            gtd = (
                title in gtd_titles
                or "has-injury-status" in classes
                or inj_text in {"gtd", "ques", "questionable", "doubtful"}
            )
            out.append(
                {
                    "name": name,
                    "position": position or None,
                    "gtd": gtd,
                }
            )
            if len(out) == 5:
                break
        return out

    def expected_starters_by_abbr(self) -> dict[str, list[dict[str, object]]]:
        """Ordered expected fives keyed by team abbreviation (WNBA/NBA)."""
        if not self.data:
            self.getDict()
        result: dict[str, list[dict[str, object]]] = {}
        for matchup_el, parsed in zip(self._matchup_cards(), self.data):
            for side, list_cls in (
                ("away", "lineup__list is-visit"),
                ("home", "lineup__list is-home"),
            ):
                abbr = parsed[side].get("abbr")
                if not abbr:
                    continue
                ul = matchup_el.find("ul", {"class": list_cls})
                starters = self._ordered_expected_starters(ul)
                if starters:
                    result[abbr] = starters
        return result

    @staticmethod
    def _players_by_title(lineup_list, titles: list[str] | str) -> set[str]:
        if lineup_list is None:
            return set()
        if isinstance(titles, str):
            titles = [titles]
        found: set[str] = set()
        for title in titles:
            for item in lineup_list.find_all("li", {"title": title}):
                classes = item.get("class") or []
                if item.a and "has-injury-status" not in classes:
                    name = item.a.get("title") or item.a.get_text(strip=True)
                    if name:
                        found.add(name)
        return found

    def getDict(self) -> list[dict]:
        """Parse matchup cards into ``self.data`` and return it."""
        self.data = []
        for matchup in self._matchup_cards():
            away_list = matchup.find("ul", {"class": "lineup__list is-visit"})
            home_list = matchup.find("ul", {"class": "lineup__list is-home"})

            away_nick = self._side_nickname(matchup, "away")
            home_nick = self._side_nickname(matchup, "home")
            away_abbr = self._resolve_abbr(matchup, "away", away_nick)
            home_abbr = self._resolve_abbr(matchup, "home", home_nick)

            if self.league == "nba":
                # Preserve historical NBA title buckets
                away_confirmed = self._players_by_title(away_list, "Very Likely To Play")
                away_gtd = self._players_by_title(
                    away_list, ["Toss Up To Play", "Likely To Play"]
                )
                home_confirmed = self._players_by_title(
                    home_list, ["Very Likely To Play", "Likely To Play"]
                )
                home_gtd = self._players_by_title(home_list, "Toss Up To Play")
            else:
                # WNBA "Expected Lineup" rows are title=Very Likely To Play
                away_confirmed = self._players_by_title(away_list, "Very Likely To Play")
                home_confirmed = self._players_by_title(home_list, "Very Likely To Play")
                away_gtd = self._players_by_title(
                    away_list, ["Toss Up To Play", "Likely To Play"]
                )
                home_gtd = self._players_by_title(
                    home_list, ["Toss Up To Play", "Likely To Play"]
                )

            away_injured = self._get_injured_players(away_list)
            home_injured = self._get_injured_players(home_list)

            self.data.append(
                {
                    "away": {
                        "team": away_nick,
                        "abbr": away_abbr,
                        "confirmed": away_confirmed,
                        "gtd": away_gtd,
                        "questionable": away_injured["questionable"],
                        "out": away_injured["out"],
                    },
                    "home": {
                        "team": home_nick,
                        "abbr": home_abbr,
                        "confirmed": home_confirmed,
                        "gtd": home_gtd,
                        "questionable": home_injured["questionable"],
                        "out": home_injured["out"],
                    },
                }
            )
        return self.data

    def updateTeamInfo(self, file_path: str | None = None) -> None:
        """Write confirmed starters / questionable lists into ``team_info.py``."""
        if not self.data:
            print("No data available. Run getDict() first.")
            return

        project_root = Path(__file__).resolve().parents[2]
        team_info_path = (
            project_root / "src" / "utils" / "team_info.py"
            if file_path is None
            else project_root / file_path
        )
        content = team_info_path.read_text(encoding="utf-8")

        updated_lineups: dict[str, list[str]] = {}
        questionable_players: dict[str, list[str]] = {}

        for matchup in self.data:
            for side in ("away", "home"):
                team_name = matchup[side]["team"]
                team_abbr = matchup[side].get("abbr") or self.team_abbreviations.get(
                    team_name
                )
                if not team_abbr:
                    print(f"Warning: no abbreviation for {team_name!r} ({side})")
                    continue

                confirmed_players = list(matchup[side]["confirmed"])
                team_questionable = list(matchup[side]["questionable"])
                if team_questionable:
                    questionable_players[team_abbr] = team_questionable

                if confirmed_players:
                    updated_lineups[team_abbr] = confirmed_players[:5]
                    if len(confirmed_players) < 5:
                        print(
                            f"Note: {team_abbr} ({team_name}) has "
                            f"{len(confirmed_players)} confirmed players"
                        )

        content = self._update_dict_in_file(
            content, self.projected_dict_name, updated_lineups
        )
        content = self._update_or_add_dict_in_file(
            content, self.questionable_dict_name, questionable_players
        )
        team_info_path.write_text(content, encoding="utf-8")

        print(f"Successfully updated {team_info_path}")
        print(
            f"[{self.league.upper()}] Updated {len(updated_lineups)} teams "
            f"→ {self.projected_dict_name}"
        )
        print(
            f"[{self.league.upper()}] Updated {len(questionable_players)} teams "
            f"→ {self.questionable_dict_name}"
        )

    def _update_dict_in_file(self, content, dict_name, updated_data):
        start_match = re.search(rf"{dict_name}\s*=\s*\{{", content)
        if not start_match:
            # Create empty dict then fill
            content = content.rstrip() + f"\n\n{dict_name} = {{\n}}\n"
            start_match = re.search(rf"{dict_name}\s*=\s*\{{", content)
            if not start_match:
                print(f"Error: Could not find or create {dict_name} in file")
                return content

        start_pos = start_match.start()
        brace_count = 0
        end_pos = start_pos
        for i, char in enumerate(content[start_pos:], start_pos):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break

        existing_content = content[start_pos:end_pos]
        existing_teams = re.findall(r'"([A-Z]{2,3})"', existing_content)

        new_dict_lines = [f"{dict_name} = {{"]
        for abbr in sorted(set(list(updated_data.keys()) + existing_teams)):
            if abbr in updated_data:
                players = updated_data[abbr]
                players_str = ", ".join(f'"{p}"' for p in players)
                new_dict_lines.append(f'    "{abbr}": [{players_str}],')
            else:
                team_match = re.search(
                    rf'"{abbr}":\s*\[(.*?)\](?:,|$)', existing_content, re.DOTALL
                )
                if team_match:
                    new_dict_lines.append(f'    "{abbr}": [{team_match.group(1)}],')
        new_dict_lines.append("}")

        before = content[:start_pos]
        after = content[end_pos:]
        return before + "\n".join(new_dict_lines) + "\n\n" + after.lstrip()

    def _update_or_add_dict_in_file(self, content, dict_name, data):
        start_match = re.search(rf"{dict_name}\s*=\s*\{{", content)
        new_dict_lines = [f"{dict_name} = {{"]
        for abbr in sorted(data.keys()):
            players_str = ", ".join(f'"{p}"' for p in data[abbr])
            new_dict_lines.append(f'    "{abbr}": [{players_str}],')
        new_dict_lines.append("}")
        block = "\n".join(new_dict_lines)

        if start_match:
            start_pos = start_match.start()
            brace_count = 0
            end_pos = start_pos
            for i, char in enumerate(content[start_pos:], start_pos):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            return content[:start_pos] + block + "\n\n" + content[end_pos:].lstrip()

        return content.rstrip() + f"\n\n# Questionable / may not play ({self.league})\n{block}\n"

    def getQuestionablePlayers(self) -> dict[str, list[str]]:
        if not self.data:
            print("No data available. Run getDict() first.")
            return {}
        out: dict[str, list[str]] = {}
        for matchup in self.data:
            for side in ("away", "home"):
                abbr = matchup[side].get("abbr") or self.team_abbreviations.get(
                    matchup[side]["team"]
                )
                if abbr and matchup[side]["questionable"]:
                    out[abbr] = list(matchup[side]["questionable"])
        return out

    def getOutPlayers(self) -> dict[str, list[str]]:
        if not self.data:
            print("No data available. Run getDict() first.")
            return {}
        out: dict[str, list[str]] = {}
        for matchup in self.data:
            for side in ("away", "home"):
                abbr = matchup[side].get("abbr") or self.team_abbreviations.get(
                    matchup[side]["team"]
                )
                if abbr and matchup[side]["out"]:
                    out[abbr] = list(matchup[side]["out"])
        return out

    def debugPrintMatchupStructure(self) -> None:
        cards = self._matchup_cards()
        if not cards:
            print("No matchup cards found")
            return
        print("=== First Matchup HTML Structure ===")
        print(cards[0].prettify()[:5000])


class NBADailyLineups(DailyLineups):
    def __init__(self, url: str | None = None):
        super().__init__(league="nba", url=url)


class WNBADailyLineups(DailyLineups):
    def __init__(self, url: str | None = None):
        super().__init__(league="wnba", url=url)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Scrape RotoWire daily lineups")
    p.add_argument("--league", choices=("nba", "wnba", "all"), default="wnba")
    p.add_argument("--update", action="store_true", help="Write team_info.py")
    p.add_argument("--print", dest="do_print", action="store_true")
    args = p.parse_args()

    leagues = ("nba", "wnba") if args.league == "all" else (args.league,)
    for league in leagues:
        scraper = DailyLineups(league=league)
        scraper.getDict()
        print(f"\n=== {league.upper()} ({len(scraper.data)} games) ===")
        if args.do_print:
            print(scraper)
        print("Questionable:", scraper.getQuestionablePlayers())
        print("Out:", scraper.getOutPlayers())
        if args.update:
            scraper.updateTeamInfo()
