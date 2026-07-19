import type { Book, LegCount } from "@/types/slate";

export const SLATE_LEG_COUNTS: LegCount[] = [2, 3, 5, 6];

export const BOOKS: Book[] = ["prizepicks", "underdog", "draftkings", "betr"];

export const BOOK_LABELS: Record<Book, string> = {
  prizepicks: "PrizePicks",
  underdog: "Underdog",
  draftkings: "DraftKings Pick 6",
  betr: "Betr",
};

export const LEG_LABELS: Record<LegCount, string> = {
  2: "2-Leg",
  3: "3-Leg",
  5: "5-Leg",
  6: "6-Leg",
};

export const DFS_BREAK_EVEN = 137 / (137 + 100);

/**
 * Net-profit multipliers (payout − 1) by platform display name × leg count.
 * Matches ``src.utils.slates._DFS_PLATFORM_PAYOUTS``.
 */
export const DFS_NET_PAYOUT_MULT: Record<string, Partial<Record<LegCount, number>>> = {
  PrizePicks: { 2: 2.0, 3: 4.5, 5: 19.0, 6: 36.5 },
  Underdog: { 2: 2.0, 3: 4.5, 5: 19.0, 6: 36.5 },
  "Betr DFS": { 2: 2.0, 3: 4.5, 5: 19.0, 6: 36.5 },
  Betr: { 2: 2.0, 3: 4.5, 5: 19.0, 6: 36.5 },
  "DraftKings Pick6": { 2: 2.0, 3: 4.0, 5: 19.0, 6: 24.0 },
  "DraftKings Pick 6": { 2: 2.0, 3: 4.0, 5: 19.0, 6: 24.0 },
};

export const DFS_DEFAULT_NET_PAYOUT: Record<LegCount, number> = {
  2: 2.0,
  3: 4.5,
  5: 19.0,
  6: 36.5,
};

export function dfsNetPayoutMult(bookmaker: string, nLegs: number): number {
  const table = DFS_NET_PAYOUT_MULT[bookmaker];
  const asLeg = nLegs as LegCount;
  if (table?.[asLeg] != null) return table[asLeg]!;
  return DFS_DEFAULT_NET_PAYOUT[asLeg] ?? DFS_DEFAULT_NET_PAYOUT[2];
}

export const SUPPORTED_MARKETS = new Set(["PTS", "REB", "AST"]);

export const PLAYERS_SORT_KEYS = [
  "player",
  "platform",
  "mkt",
  "line",
  "tier",
  "modelProb",
  "sharpProb",
  "consensusProb",
  "statProj",
  "minProj",
  "l5",
  "l10",
  "l15",
  "vsOppAvg",
  "oppDefRank",
] as const;

export type PlayersSortKey = (typeof PLAYERS_SORT_KEYS)[number];
