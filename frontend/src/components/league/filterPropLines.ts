import type { ApiWnbaGame, ApiWnbaPropLine } from "@/lib/api";

export type PropFilterSelection = {
  stats: Set<string>;
  sides: Set<string>;
  teams: Set<string>;
};

export type TeamFilterOption = {
  abbrev: string;
  logoUrl: string | null;
};

export function filterPropLines(
  props: ApiWnbaPropLine[],
  selection: PropFilterSelection,
): ApiWnbaPropLine[] {
  const { stats, sides, teams } = selection;
  return props.filter((row) => {
    if (stats.size > 0 && !stats.has(row.stat)) return false;
    if (sides.size > 0 && !sides.has(row.side.toLowerCase())) return false;
    if (teams.size > 0) {
      if (!row.team_abbrev || !teams.has(row.team_abbrev)) return false;
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
