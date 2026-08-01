import type { ApiWnbaGame, ApiWnbaPropLine } from "@/lib/api";

export const PROP_BOOK_OPTIONS = [
  { key: "fanduel", label: "FanDuel" },
  { key: "draftkings", label: "DraftKings" },
  { key: "caesars", label: "Caesars" },
  { key: "betmgm", label: "BetMGM" },
  { key: "pinnacle", label: "Pinnacle" },
  { key: "bet365", label: "bet365" },
  { key: "prizepicks", label: "PrizePicks" },
  { key: "underdog", label: "Underdog" },
  { key: "betr", label: "Betr" },
  { key: "novig", label: "Novig" },
  { key: "sleeper", label: "Sleeper" },
  { key: "pick6", label: "Pick6" },
] as const;

export type PropBookKey = (typeof PROP_BOOK_OPTIONS)[number]["key"];

export type PropFilterSelection = {
  stats: Set<string>;
  sides: Set<string>;
  teams: Set<string>;
  books: Set<string>;
};

export type TeamFilterOption = {
  abbrev: string;
  logoUrl: string | null;
};

function rowHasBook(row: ApiWnbaPropLine, book: string): boolean {
  const quote = row[book as PropBookKey];
  return quote != null;
}

export function filterPropLines(
  props: ApiWnbaPropLine[],
  selection: PropFilterSelection,
): ApiWnbaPropLine[] {
  const { stats, sides, teams, books } = selection;
  return props.filter((row) => {
    if (stats.size > 0 && !stats.has(row.stat)) return false;
    if (sides.size > 0 && !sides.has(row.side.toLowerCase())) return false;
    if (teams.size > 0) {
      if (!row.team_abbrev || !teams.has(row.team_abbrev)) return false;
    }
    if (books.size > 0) {
      let hasSelected = false;
      for (const book of books) {
        if (rowHasBook(row, book)) {
          hasSelected = true;
          break;
        }
      }
      if (!hasSelected) return false;
    }
    return true;
  });
}

export function collectStatOptions(props: ApiWnbaPropLine[]): string[] {
  return [...new Set(props.map((p) => p.stat).filter(Boolean))].sort((a, b) =>
    a.localeCompare(b),
  );
}

export function collectTeamOptions(props: ApiWnbaPropLine[]): TeamFilterOption[] {
  const byAbbrev = new Map<string, string | null>();
  for (const row of props) {
    if (!row.team_abbrev) continue;
    if (!byAbbrev.has(row.team_abbrev) || !byAbbrev.get(row.team_abbrev)) {
      byAbbrev.set(row.team_abbrev, row.logo_url);
    }
  }
  return [...byAbbrev.entries()]
    .map(([abbrev, logoUrl]) => ({ abbrev, logoUrl }))
    .sort((a, b) => a.abbrev.localeCompare(b.abbrev));
}

export function excludePropsFromFinalGames(
  props: ApiWnbaPropLine[],
  games: ApiWnbaGame[] | undefined | null,
): ApiWnbaPropLine[] {
  if (!games || games.length === 0) return props;

  const finalTeams = new Set<string>();
  for (const g of games) {
    if (g.status !== "final") continue;
    if (g.home?.abbrev) finalTeams.add(g.home.abbrev);
    if (g.away?.abbrev) finalTeams.add(g.away.abbrev);
  }
  if (finalTeams.size === 0) return props;

  return props.filter(
    (row) => !row.team_abbrev || !finalTeams.has(row.team_abbrev),
  );
}

/** Add one calendar day to a YYYY-MM-DD string. */
export function nextEtDate(dateIso: string): string {
  const [y, m, d] = dateIso.split("-").map(Number);
  const utc = new Date(Date.UTC(y!, (m ?? 1) - 1, d ?? 1));
  utc.setUTCDate(utc.getUTCDate() + 1);
  return utc.toISOString().slice(0, 10);
}

/**
 * Tip date in America/New_York for a commence_time ISO string.
 * Returns YYYY-MM-DD or null when unparsable.
 */
export function tipEtDate(commenceTime: string | null | undefined): string | null {
  if (!commenceTime) return null;
  const parsed = new Date(commenceTime);
  if (Number.isNaN(parsed.getTime())) return null;
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(parsed);
}

/**
 * Drop props for finished / past games.
 *
 * - Teams already final on the scoreboard are removed.
 * - Tips whose ET calendar date is before the scoreboard slate date are removed
 *   (covers prior-day leftovers not on today's board).
 * Upcoming and in-progress tips are kept.
 */
export function excludePastGameProps(
  props: ApiWnbaPropLine[],
  games: ApiWnbaGame[] | undefined | null,
  scoreboardDate: string | null | undefined,
): ApiWnbaPropLine[] {
  let out = excludePropsFromFinalGames(props, games);
  if (!scoreboardDate) return out;

  return out.filter((row) => {
    const tip = tipEtDate(row.commence_time);
    if (!tip) return true;
    return tip >= scoreboardDate;
  });
}
