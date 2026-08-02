import type { HomeLeague } from "./types";

export function gameDetailHref(game: {
  league: HomeLeague;
  espnEventId?: string | null;
  mlbGamePk?: string | null;
}): string | null {
  if (game.league === "mlb" && game.mlbGamePk) {
    return `/mlb/games/${game.mlbGamePk}`;
  }
  if (game.espnEventId) {
    return `/games/${game.espnEventId}`;
  }
  return null;
}
