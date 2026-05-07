import copy
import requests
import pandas as pd
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

# Project root (parent of src/) so `scripts` package resolves
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

_DEFAULT_UNDERDOG_DIR = os.path.join(_ROOT, 'data', 'props', 'underdogs')
_OUTPUT_TZ = ZoneInfo("America/Los_Angeles")

_UNDERDOG_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

# Used when config.json is missing; optional file can override keys
_DEFAULT_UNDERDOG_CONFIG = {
    "ud_pickem_url": "https://api.underdogfantasy.com/beta/v5/over_under_lines",
    "headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://app.underdogfantasy.com/",
    },
}


def _underdog_output_filename() -> str:
    """underdog_2026-05-06_213045.json in Pacific time."""
    d = datetime.now(_OUTPUT_TZ)
    return d.strftime("underdog_%Y-%m-%d_%H%M%S.json")


def _resolve_underdog_output_path() -> str:
    """
    Always data/props/underdogs under project root + timestamped .json.
    Optional UNDERDOG_OUTPUT=/abs/path/file.json overrides to a fixed file path.
    """
    out = os.environ.get("UNDERDOG_OUTPUT", "").strip()
    if out and out.lower().endswith(".json"):
        expanded = os.path.expanduser(out)
        if not expanded.endswith(("/", "\\")) and not os.path.isdir(expanded):
            return expanded
    return os.path.join(_DEFAULT_UNDERDOG_DIR, _underdog_output_filename())


class UnderdogScraper:
    def __init__(self):
        self.config = None
        self.underdog_props = None

        self.directory = _resolve_underdog_output_path()

        print(f"Using file path: {self.directory}")
        self.load_config()

    def load_config(self):
        cfg = copy.deepcopy(_DEFAULT_UNDERDOG_CONFIG)
        if os.path.isfile(_UNDERDOG_CONFIG_PATH):
            with open(_UNDERDOG_CONFIG_PATH, encoding="utf-8-sig") as json_file:
                user_cfg = json.load(json_file)
            cfg.update(user_cfg)
            if "headers" in user_cfg:
                cfg["headers"] = {
                    **_DEFAULT_UNDERDOG_CONFIG["headers"],
                    **user_cfg["headers"],
                }
        self.config = cfg

    def fetch_data(self):
        ud_pickem_response = requests.get(self.config["ud_pickem_url"], headers=self.config["headers"])

        if ud_pickem_response.status_code != 200:
            raise Exception("Request failed")

        pickem_data = json.loads(ud_pickem_response.text)

        return pickem_data

    def combine_data(self, pickem_data):
        players = pd.DataFrame(pickem_data["players"])
        appearances = pd.DataFrame(pickem_data["appearances"])
        # games = pd.DataFrame(pickem_data["games"])
        over_under_lines = pd.DataFrame(pickem_data["over_under_lines"])

        return players, appearances, over_under_lines
    
    def apply_name_corrections(self, df):
        name_corrections = {
            # ... If you're working with other data sets use this dictionary match names
        }
        df["full_name"] = df["full_name"].map(name_corrections).fillna(df["full_name"])
        return df

    def process_data(self, players, appearances, over_under_lines):
        players = players.rename(columns={"id": "player_id"})
        appearances = appearances.rename(columns={"id": "appearance_id"})

        player_appearances = players.merge(appearances, on=["player_id", "position_id", "team_id"], how="left")

        over_under_lines = over_under_lines.reset_index(drop=True)
        over_under_lines_expanded = over_under_lines.explode("options")

        options_df = pd.json_normalize(over_under_lines_expanded["options"]).reset_index(drop=True)
        left_parts = over_under_lines_expanded.drop("options", axis=1).reset_index(drop=True)

        overlap = set(left_parts.columns) & set(options_df.columns)
        if overlap:
            options_df = options_df.rename(columns={c: f"option_{c}" for c in overlap})

        over_under_lines_expanded = pd.concat([left_parts, options_df], axis=1)

        over_under_lines_expanded["appearance_id"] = over_under_lines_expanded["over_under"].apply(lambda x: x["appearance_stat"]["appearance_id"])
        over_under_lines_expanded["stat_name"] = over_under_lines_expanded["over_under"].apply(lambda x: x["appearance_stat"]["stat"])

        columns_to_remove = ['expires_at', 'live_event', 'live_event_stat']
        over_under_lines_expanded = over_under_lines_expanded.drop(columns=columns_to_remove, errors='ignore')

        over_under_lines_expanded["choice"] = over_under_lines_expanded["choice"].map({"lower": "under", "higher": "over"}).fillna(over_under_lines_expanded["choice"])

        underdog_props = player_appearances.merge(over_under_lines_expanded, on="appearance_id", how="left", suffixes=("", "_over_under"))
        underdog_props["full_name"] = underdog_props["first_name"] + " " + underdog_props["last_name"]

        underdog_props = self.apply_name_corrections(underdog_props)

        if underdog_props.columns.duplicated().any():
            underdog_props = underdog_props.loc[:, ~underdog_props.columns.duplicated()].copy()

        return underdog_props

    def filter_data(self, df):
        # df = df[df["sport_id"].isin(["MLB"])]
        suspend_mask = pd.Series(True, index=df.index)
        for col in ("status", "status_over_under"):
            if col in df.columns:
                suspend_mask &= df[col].astype(str) != "suspended"

        df = df[suspend_mask]

        columns_to_remove = ['country', 'image_url', 'badges', 'lineup_status_id', 'match_id', 'match_type', 'over_under', 'rank', 'status', 'status_over_under']
        df = df.drop(columns=columns_to_remove, errors='ignore')
        df = df.reset_index(drop=True)

        return df

    def scrape(self):
        all_pickem_data = self.fetch_data()
        players, appearances, over_under_lines = self.combine_data(all_pickem_data)
        processed_props = self.process_data(players, appearances, over_under_lines)
        self.underdog_props = self.filter_data(processed_props)

        directory_path = os.path.dirname(self.directory)
        if directory_path and not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)

        self.underdog_props.to_json(
            self.directory,
            orient='records',
            indent=2,
            date_format='iso',
            force_ascii=False,
        )
        print(f"✓ Successfully saved Underdog props to {self.directory}")


if __name__ == "__main__":
    print("Starting Underdog Scraper...")
    try:
        scraper = UnderdogScraper()
        scraper.scrape()
        n = len(scraper.underdog_props) if scraper.underdog_props is not None else 0
        print(f"\n✓ Scraping complete! {n} rows.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
