import type { ApiWnbaOddsGame } from "@/lib/api";
import type { MatchupGame, MatchupOdds } from "./types";

/** Align ESPN tricodes with odds / stats.wnba.com spellings. */
const ABBREV_ALIASES: Record<string, string> = {
  GS: "GSV",
  LA: "LAS",
  LV: "LVA",
  NY: "NYL",
  PHX: "PHO",
  POR: "PDX",
  WSH: "WAS",
};

function canonicalAbbrev(abbrev: string): string {
  const upper = abbrev.trim().toUpperCase();
  return ABBREV_ALIASES[upper] ?? upper;
}

function oddsKey(homeAbbrev: string, awayAbbrev: string): string {
  return `${canonicalAbbrev(awayAbbrev)}@${canonicalAbbrev(homeAbbrev)}`;
}

function toMatchupOdds(game: ApiWnbaOddsGame): MatchupOdds | null {
  const spreadTeamAbbrev = game.spread_team_abbrev
    ? canonicalAbbrev(game.spread_team_abbrev)
    : null;
  const spreadLine = game.spread_line ?? null;
  const total = game.total ?? null;
  if (spreadLine == null && total == null) {
    return null;
  }
  return { spreadTeamAbbrev, spreadLine, total };
}

export function formatOddsPill(odds: MatchupOdds): string | null {
  const parts: string[] = [];
  if (odds.spreadLine != null && odds.spreadTeamAbbrev) {
    parts.push(`Spread: ${odds.spreadTeamAbbrev} ${odds.spreadLine}`);
  }
  if (odds.total != null) {
    parts.push(`Total: ${odds.total}`);
  }
  return parts.length > 0 ? parts.join(" · ") : null;
}

export function mergeMatchupOdds(
  games: MatchupGame[],
  oddsGames: ApiWnbaOddsGame[] | undefined,
  slateDate?: string,
): MatchupGame[] {
  if (!oddsGames || oddsGames.length === 0) {
    return games.map((game) => ({ ...game, odds: game.odds ?? null }));
  }

  const byDated = new Map<string, MatchupOdds>();
  const byUndated = new Map<string, MatchupOdds>();

  for (const row of oddsGames) {
    const odds = toMatchupOdds(row);
    if (!odds) continue;
    const key = oddsKey(row.home_abbrev, row.away_abbrev);
    if (row.game_date) {
      byDated.set(`${row.game_date}|${key}`, odds);
    } else {
      byUndated.set(key, odds);
    }
  }

  return games.map((game) => {
    const key = oddsKey(game.home.abbrev, game.away.abbrev);
    let odds: MatchupOdds | null = null;
    if (slateDate) {
      odds = byDated.get(`${slateDate}|${key}`) ?? byUndated.get(key) ?? null;
    } else {
      odds = byUndated.get(key) ?? null;
      if (!odds) {
        // abbrev-only legacy: accept any dated row for this matchup
        for (const [datedKey, value] of byDated) {
          if (datedKey.endsWith(`|${key}`)) {
            odds = value;
            break;
          }
        }
      }
    }
    return { ...game, odds };
  });
}
