import type { ApiLiveSlatesResponse } from "@/types/api";
import type { EnrichedPick, FlatParlayRow, ParlayLegRaw } from "@/types/slate";

function pickKey(player: string | undefined | null, market: string | undefined | null): string {
  return `${String(player || "").trim().toLowerCase()}|${String(market || "").trim().toUpperCase()}`;
}

function contextScore(p: EnrichedPick): number {
  let s = 0;
  if (p.vs_opp?.avg_stat != null) s += 2;
  if (p.vs_opp?.n_games != null && Number(p.vs_opp.n_games) > 0) s += 1;
  if (p.game_context?.opp_def_rating_rank != null) s += 2;
  if (p.game_context?.opp_pace_rank != null) s += 1;
  if (p.opponent_abbr) s += 1;
  return s;
}

/** Prefer live-props rows that carry matchup / opp-rank context. */
function indexLiveProps(picks: EnrichedPick[]): Map<string, EnrichedPick> {
  const idx = new Map<string, EnrichedPick>();
  for (const p of picks) {
    const key = pickKey(p.player ?? p.display_name, p.market);
    if (key.startsWith("|")) continue;
    const prev = idx.get(key);
    if (!prev || contextScore(p) > contextScore(prev)) idx.set(key, p);
  }
  return idx;
}

function fillLegFromLiveProps(leg: ParlayLegRaw, src: EnrichedPick): ParlayLegRaw {
  const vs = { ...(leg.vs_opp || {}) };
  const gc = { ...(leg.game_context || {}) };

  if (vs.avg_stat == null && src.vs_opp?.avg_stat != null) {
    vs.avg_stat = src.vs_opp.avg_stat;
  }
  if (
    (vs.n_games == null || Number(vs.n_games) === 0) &&
    src.vs_opp?.n_games != null &&
    Number(src.vs_opp.n_games) > 0
  ) {
    vs.n_games = src.vs_opp.n_games;
  }

  if (gc.opp_def_rating_rank == null && src.game_context?.opp_def_rating_rank != null) {
    gc.opp_def_rating_rank = src.game_context.opp_def_rating_rank;
  }
  if (gc.opp_pace_rank == null && src.game_context?.opp_pace_rank != null) {
    gc.opp_pace_rank = src.game_context.opp_pace_rank;
  }
  if (gc.spread == null && src.game_context?.spread != null) {
    gc.spread = src.game_context.spread;
  }
  if (gc.game_total == null && src.game_context?.game_total != null) {
    gc.game_total = src.game_context.game_total;
  }

  return {
    ...leg,
    team_abbr: leg.team_abbr || src.team_abbr,
    opponent_abbr: leg.opponent_abbr || src.opponent_abbr,
    opponent: leg.opponent || src.opponent_abbr,
    vs_opp: vs,
    game_context: gc,
  };
}

/**
 * Overlay All Players (live-props) matchup / opp-rank fields onto slate parlay legs
 * when the enrich path left them empty.
 */
export function enrichSlatesFromLiveProps(
  response: ApiLiveSlatesResponse,
  picks: EnrichedPick[],
): ApiLiveSlatesResponse {
  if (!picks.length) return response;
  const idx = indexLiveProps(picks);
  if (!idx.size) return response;

  const slates: ApiLiveSlatesResponse["slates"] = {};
  for (const [nLegs, byBook] of Object.entries(response.slates || {})) {
    slates[nLegs] = {};
    for (const [book, parlays] of Object.entries(byBook || {})) {
      slates[nLegs][book] = (parlays || []).map((parlay: FlatParlayRow) => {
        const legs = parlay.LEGS;
        if (!Array.isArray(legs) || !legs.length) return parlay;
        const nextLegs = legs.map((leg) => {
          const key = pickKey(leg.player ?? leg.display_name, leg.market);
          const src = idx.get(key);
          return src ? fillLegFromLiveProps(leg, src) : leg;
        });
        return { ...parlay, LEGS: nextLegs };
      });
    }
  }

  return { ...response, slates };
}
