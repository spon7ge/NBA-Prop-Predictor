import type { ApiLivePropPick, ApiLivePropsResponse } from "@/types/api";
import type { EnrichedPick } from "@/types/slate";

/** Map Odds-API / DB bookmaker keys → All Players platform filter labels. */
const PLATFORM_LABEL: Record<string, string> = {
  prizepicks: "PrizePicks",
  underdog: "Underdog",
  draftkings: "DraftKings Pick6",
  "draftkings pick6": "DraftKings Pick6",
  "draftkings pick 6": "DraftKings Pick6",
  betr: "Betr DFS",
  "betr dfs": "Betr DFS",
};

function normalizePlatform(raw: string | undefined | null): string {
  const key = String(raw || "").trim().toLowerCase();
  return PLATFORM_LABEL[key] || String(raw || "").trim();
}

function num(v: number | null | undefined): number | undefined {
  return v == null || Number.isNaN(Number(v)) ? undefined : Number(v);
}

/** Map GET /api/live-props picks → EnrichedPick for AllPlayersView. */
export function mapLivePropsToEnrichedPicks(
  resp: ApiLivePropsResponse,
): EnrichedPick[] {
  return (resp.picks || [])
    .filter((p) => p.line != null && Number.isFinite(Number(p.line)))
    .map((p: ApiLivePropPick): EnrichedPick => ({
    player: p.player,
    display_name: p.player,
    team_abbr: p.team_abbr ?? undefined,
    opponent_abbr: p.opponent_abbr ?? undefined,
    is_home: p.is_home ?? undefined,
    market: p.market,
    dfs_line: num(p.line),
    platform: normalizePlatform(p.platform),
    model: {
      lean: p.model?.lean ?? undefined,
      stat_q50: num(p.model?.stat_q50),
      min_q50: num(p.model?.min_q50),
      p_over: num(p.model?.p_over),
    },
    game_context: {
      spread: num(p.game_context?.team_spread),
      game_total: num(p.game_context?.game_total),
      opp_def_rating_rank: num(p.game_context?.opp_def_rating_rank) as
        | number
        | undefined,
      opp_pace: num(p.game_context?.opp_pace),
    },
    form: {
      over_l5: num(p.form?.over_l5),
      over_l10: num(p.form?.over_l10),
      over_l15: num(p.form?.over_l15),
    },
    vs_opp: {
      avg_stat: num(p.vs_opp?.avg_stat),
      n_games: num(p.vs_opp?.n_games) as number | undefined,
    },
  }));
}
