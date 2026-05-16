import time
import os
import random
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from nba_api.stats.endpoints import playbyplayv3

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in (
        'rate limit', 'too many requests', '429',
        'timeout', 'timed out', 'connection', 'read timed out',
    ))


def _call_with_retry(fn, *args, label: str = '', max_retries: int = 5,
                     base_delay: float = 1.0, max_delay: float = 60.0, **kwargs):
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not _is_rate_limit_error(e) or attempt >= max_retries:
                raise
            sleep_for = min(max_delay, base_delay * (2 ** attempt))
            sleep_for += random.uniform(0, 0.5 * sleep_for)
            logger.warning(
                "Rate limit / transient error on %s (attempt %d/%d): %s — "
                "retrying in %.1fs",
                label or fn.__name__, attempt + 1, max_retries, e, sleep_for,
            )
            time.sleep(sleep_for)
            attempt += 1


class NBAPlayByPlay:
    """
    Fetches play-by-play data for a list of game IDs using playbyplayv3.

    Usage:
        pbp = NBAPlayByPlay(game_ids).fetch().build()
        df  = pbp.get_df()
    """

    def __init__(self, game_ids):
        self.game_ids = [str(gid).zfill(10) for gid in game_ids]
        self._frames = []
        self.final = None

    # ── Fetch ─────────────────────────────────────────────────────────────────

    def fetch(
        self,
        delay: float = 0.3,
        workers: int = 8,
        batch_size: int = 100,
        checkpoint_path: str = 'pbp_checkpoint.csv',
        run_all_batches: bool = True,
    ) -> 'NBAPlayByPlay':
        total = len(self.game_ids)

        if os.path.exists(checkpoint_path):
            checkpoint = pd.read_csv(checkpoint_path, dtype={'gameId': str})
            checkpoint = checkpoint[checkpoint['gameId'].isin(self.game_ids)]
            done_ids = set(checkpoint['gameId'].unique())
            remaining = [gid for gid in self.game_ids if gid not in done_ids]
            self._frames = [checkpoint] if not checkpoint.empty else []
            print(f"✓ Checkpoint loaded — {len(done_ids)}/{total} games done, {len(remaining)} remaining")
        else:
            remaining = list(self.game_ids)
            self._frames = []
            print(f"Fetching play-by-play for {total} games (batch size: {batch_size})...")

        if not remaining:
            print("✓ All games already fetched from checkpoint")
            return self

        failed = []
        batches = [remaining[i:i + batch_size] for i in range(0, len(remaining), batch_size)]

        for batch_num, batch in enumerate(batches, 1):
            print(f"\n── Batch {batch_num}/{len(batches)} ({len(batch)} games) ──")
            batch_frames = []

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(self._fetch_one, gid, delay): gid for gid in batch}

                for i, future in enumerate(as_completed(futures), 1):
                    game_id = futures[future]
                    try:
                        result_df = future.result()
                        if result_df is None:
                            failed.append(game_id)
                        else:
                            batch_frames.append(result_df)
                    except Exception as e:
                        failed.append(game_id)
                        logger.warning("Skipped game %s after retries: %s", game_id, e)

                    if i % 25 == 0:
                        print(f"  … {i}/{len(batch)} games in batch")

            if batch_frames:
                self._frames.extend(batch_frames)
                checkpoint_so_far = pd.concat(self._frames, ignore_index=True)
                checkpoint_so_far.to_csv(checkpoint_path, index=False)
                print(f"  ✓ Batch {batch_num} done — checkpoint saved ({len(checkpoint_so_far)} rows total)")

            if not run_all_batches:
                print(f"  → Stopping after batch {batch_num} (run_all_batches=False).")
                break

        if failed:
            print(f"\n  ✗ {len(failed)} games failed: {failed}")

        return self

    def _fetch_one(self, game_id: str, delay: float):
        time.sleep(delay)

        def _call():
            return playbyplayv3.PlayByPlayV3(game_id=game_id).get_data_frames()[0]

        df = _call_with_retry(_call, label=f"game {game_id}")

        if df is None or df.empty:
            logger.warning("Skipping game %s: no PBP data returned", game_id)
            return None

        return df

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self) -> 'NBAPlayByPlay':
        if not self._frames:
            raise RuntimeError("Call .fetch() before .build(), or no data was returned")
        self.final = pd.concat(self._frames, ignore_index=True)
        print(f"✓ Built — shape: {self.final.shape}")
        return self

    # ── Access ────────────────────────────────────────────────────────────────

    def get_df(self) -> pd.DataFrame:
        if self.final is None:
            raise RuntimeError("Call .fetch().build() first")
        return self.final

    def filter_game(self, game_id: str) -> pd.DataFrame:
        return self.final[self.final['gameId'] == str(game_id).zfill(10)]

    def filter_player(self, player_name: str) -> pd.DataFrame:
        return self.final[self.final['playerName'].str.lower() == player_name.lower()]

    def filter_action(self, action_type: str) -> pd.DataFrame:
        return self.final[self.final['actionType'].str.lower() == action_type.lower()]
