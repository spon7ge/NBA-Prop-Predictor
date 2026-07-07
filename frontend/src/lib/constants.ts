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
