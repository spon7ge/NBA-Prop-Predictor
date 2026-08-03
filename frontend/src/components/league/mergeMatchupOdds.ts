import type { ApiMatchupOddsGame } from "@/lib/api";
import type { MatchupGame, MatchupOdds } from "./types";

/** Align ESPN tricodes with odds / stats.wnba.com spellings (WNBA only). */
const ABBREV_ALIASES: Record<string, string> = {
  GS: "GSV",
  LA: "LAS",
  LV: "LVA",
  NY: "NYL",
  PHX: "PHO",
  POR: "PDX",
  CONN: "CON", // Sharp CONN vs ESPN/stats CON (Connecticut Sun)
  WSH: "WAS",
};

function canonicalAbbrev(abbrev: string, wnbaAliases: boolean): string {
  const upper = abbrev.trim().toUpperCase();
  if (!wnbaAliases) return upper;
  return ABBREV_ALIASES[upper] ?? upper;
}

function oddsKey(
  homeAbbrev: string,
  awayAbbrev: string,
  wnbaAliases: boolean,
): string {
  return `${canonicalAbbrev(awayAbbrev, wnbaAliases)}@${canonicalAbbrev(homeAbbrev, wnbaAliases)}`;
}

function toMatchupOdds(
  game: ApiMatchupOddsGame,
  wnbaAliases: boolean,
): MatchupOdds | null {
  const spreadTeamAbbrev = game.spread_team_abbrev
    ? canonicalAbbrev(game.spread_team_abbrev, wnbaAliases)
    : null;
  const spreadLine = game.spread_line ?? null;
  const total = game.total ?? null;
  if (spreadLine == null && total == null) {
    return null;
  }
  const sportsbook = game.sportsbook?.trim() || null;
  return { spreadTeamAbbrev, spreadLine, total, sportsbook };
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
  oddsGames: ApiMatchupOddsGame[] | undefined,
  slateDate?: string,
  options?: { wnbaAliases?: boolean },
): MatchupGame[] {
  const wnbaAliases = options?.wnbaAliases ?? true;
  if (!oddsGames || oddsGames.length === 0) {
    return games.map((game) => ({ ...game, odds: game.odds ?? null }));
  }

  const byDated = new Map<string, MatchupOdds>();
  const byUndated = new Map<string, MatchupOdds>();

  for (const row of oddsGames) {
    const odds = toMatchupOdds(row, wnbaAliases);
    if (!odds) continue;
    const key = oddsKey(row.home_abbrev, row.away_abbrev, wnbaAliases);
    if (row.game_date) {
      byDated.set(`${row.game_date}|${key}`, odds);
    } else {
      byUndated.set(key, odds);
    }
  }

  return games.map((game) => {
    const key = oddsKey(game.home.abbrev, game.away.abbrev, wnbaAliases);
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
