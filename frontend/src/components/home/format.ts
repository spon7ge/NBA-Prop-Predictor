/** Skeleton placeholders shown when there are no live games yet. */
export const LIVE_NOW_SKELETON_COUNT = 3;

/**
 * Formats the LIVE NOW subtitle count.
 * Rules: non-negative integer; singular "game" when count === 1.
 */
export function formatGamesInProgress(count: number): string {
  if (!Number.isFinite(count) || count < 0) {
    throw new Error("Game count must be a non-negative finite number");
  }
  const n = Math.floor(count);
  return `${n} ${n === 1 ? "game" : "games"} in progress`;
}

/**
 * Normalizes the optional games prop so callers can pass undefined or a list.
 */
export function normalizeLiveGames<T>(games: T[] | undefined | null): T[] {
  if (games == null) return [];
  if (!Array.isArray(games)) {
    throw new Error("games must be an array when provided");
  }
  return games;
}
