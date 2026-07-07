import type { ApiGameSlate, ApiMLPrediction, ApiPropLine } from "@/types/api";
import type { EnrichedPick, Tier } from "@/types/slate";
import { SUPPORTED_MARKETS } from "@/lib/constants";

const MARKET_LABELS: Record<string, string> = {
  player_points: "PTS",
  player_rebounds: "REB",
  player_assists: "AST",
};

const STAT_PROP: Record<string, string> = {
  PTS: "ppm",
  REB: "rpm",
  AST: "apm",
};

const SHARP_BOOKS = ["pinnacle"];

function normalizeMarket(category: string): string | null {
  const key = category.toLowerCase().replace(/\s+/g, "_");
  return MARKET_LABELS[key] ?? null;
}

function bookToPlatform(bookmaker: string): string {
  const b = bookmaker.toLowerCase();
  if (b.includes("prize")) return "PrizePicks";
  if (b.includes("underdog")) return "Underdog";
  if (b.includes("draft")) return "DraftKings Pick6";
  if (b.includes("betr")) return "Betr DFS";
  return bookmaker;
}

function isDfsBook(bookmaker: string, source?: string): boolean {
  if (source === "dfs") return true;
  const b = bookmaker.toLowerCase();
  return (
    b.includes("prize") ||
    b.includes("underdog") ||
    (b.includes("draft") && b.includes("pick")) ||
    b.includes("betr")
  );
}

function isUsBook(bookmaker: string, source?: string): boolean {
  if (source === "us") return true;
  const b = bookmaker.toLowerCase();
  return (
    b.includes("pinnacle") ||
    b.includes("fanduel") ||
    b.includes("draftkings") ||
    b.includes("betmgm") ||
    b.includes("caesars") ||
    b.includes("bet365")
  );
}

function americanToImplied(odds: number): number {
  if (odds > 0) return 100 / (odds + 100);
  return Math.abs(odds) / (Math.abs(odds) + 100);
}

function noVigOver(overOdds: number, underOdds: number): number {
  const pOver = americanToImplied(overOdds);
  const pUnder = americanToImplied(underOdds);
  const total = pOver + pUnder;
  return total > 0 ? pOver / total : 0.5;
}

function estimatePOver(projection: number, line: number): number {
  const diff = projection - line;
  return Math.max(0.05, Math.min(0.95, 0.5 + diff * 0.08));
}

function inferOpponent(prop: ApiPropLine): { opp: string | null; isHome: boolean } {
  const team = prop.player_team_abbrev;
  const home = prop.home_team_abbrev;
  const away = prop.away_team_abbrev;
  if (!team || !home || !away) return { opp: null, isHome: false };
  if (team === home) return { opp: away, isHome: true };
  if (team === away) return { opp: home, isHome: false };
  return { opp: null, isHome: false };
}

function buildPredictionIndex(
  predictions: ApiMLPrediction[],
): Map<string, Record<string, number>> {
  const idx = new Map<string, Record<string, number>>();
  for (const p of predictions) {
    const key = `${p.player_id}:${p.game_id}`;
    const bucket = idx.get(key) ?? {};
    bucket[p.prop] = p.prediction;
    idx.set(key, bucket);
  }
  return idx;
}

interface OddsPair {
  over?: number;
  under?: number;
}

function buildOddsPairs(props: ApiPropLine[]): Map<string, OddsPair> {
  const pairs = new Map<string, OddsPair>();
  for (const p of props) {
    const key = `${p.player_id ?? p.player_name}:${p.bookmaker}:${p.market_category}:${p.line}`;
    const entry = pairs.get(key) ?? {};
    if (p.side.toLowerCase() === "over") entry.over = p.odds;
    else entry.under = p.odds;
    pairs.set(key, entry);
  }
  return pairs;
}

function pickTier(modelLean: string | undefined, sharpLean: string | undefined): Tier {
  if (!modelLean) return "no_model";
  if (!sharpLean) return "no_model";
  return modelLean === sharpLean ? "sharp_verified" : "conflict";
}

function statRateForMarket(market: string, prop: ApiPropLine): number | undefined {
  if (market === "PTS") return prop.pts_per_min_roll10 ?? prop.pts_per_min_roll5;
  if (market === "REB") return prop.reb_per_min_roll5;
  if (market === "AST") return prop.ast_per_min_roll5;
  return undefined;
}

function buildModelFields(
  market: string,
  line: number,
  preds: Record<string, number> | undefined,
  minRoll10?: number,
): EnrichedPick["model"] {
  const minQ50 = preds?.min ?? minRoll10;
  const statProp = STAT_PROP[market];
  const rate = statProp && preds ? preds[statProp] : undefined;
  let statQ50: number | undefined;
  if (rate != null && minQ50 != null) statQ50 = rate * minQ50;
  else if (rate != null && minRoll10 != null) statQ50 = rate * minRoll10;

  if (statQ50 == null) return undefined;

  const pOver = estimatePOver(statQ50, line);
  const lean = statQ50 >= line ? "OVER" : "UNDER";
  return {
    lean,
    stat_q50: statQ50,
    min_q50: minQ50,
    p_over: pOver,
  };
}

function findGameId(
  prop: ApiPropLine,
  predictions: ApiMLPrediction[],
): string | undefined {
  if (!prop.player_id) return undefined;
  const match = predictions.find(
    (p) =>
      p.player_id === prop.player_id &&
      (!prop.game_date || !p.game_date || p.game_date === prop.game_date),
  );
  return match?.game_id;
}

function buildSharpConsensus(
  prop: ApiPropLine,
  market: string,
  usProps: ApiPropLine[],
  oddsPairs: Map<string, OddsPair>,
): { sharp?: EnrichedPick["sharp"]; consensus?: EnrichedPick["consensus"] } {
  const sameLine = usProps.filter(
    (u) =>
      u.player_id === prop.player_id &&
      normalizeMarket(u.market_category) === market &&
      u.line === prop.line,
  );

  let sharp: EnrichedPick["sharp"];
  for (const book of SHARP_BOOKS) {
    const sharpProp = sameLine.find((u) => u.bookmaker.toLowerCase().includes(book));
    if (!sharpProp) continue;
    const pairKey = `${sharpProp.player_id ?? sharpProp.player_name}:${sharpProp.bookmaker}:${sharpProp.market_category}:${sharpProp.line}`;
    const pair = oddsPairs.get(pairKey);
    if (pair?.over != null && pair?.under != null) {
      const nv = noVigOver(pair.over, pair.under);
      sharp = {
        lean: nv >= 0.5 ? "OVER" : "UNDER",
        no_vig_over: nv,
      };
      break;
    }
  }

  const noVigValues: number[] = [];
  const seenBooks = new Set<string>();
  for (const u of sameLine) {
    if (seenBooks.has(u.bookmaker)) continue;
    const pairKey = `${u.player_id ?? u.player_name}:${u.bookmaker}:${u.market_category}:${u.line}`;
    const pair = oddsPairs.get(pairKey);
    if (pair?.over != null && pair?.under != null) {
      noVigValues.push(noVigOver(pair.over, pair.under));
      seenBooks.add(u.bookmaker);
    }
  }

  const consensus =
    noVigValues.length > 0
      ? {
          mean_no_vig_over_same_line:
            noVigValues.reduce((a, b) => a + b, 0) / noVigValues.length,
          n_books_same_line: noVigValues.length,
        }
      : undefined;

  return { sharp, consensus };
}

export function mapSlateToEnrichedPicks(slate: ApiGameSlate): EnrichedPick[] {
  const predIdx = buildPredictionIndex(slate.predictions);
  const oddsPairs = buildOddsPairs(slate.props);
  const usProps = slate.props.filter((p) => isUsBook(p.bookmaker, p.prop_source));

  const dfsOverProps = slate.props.filter(
    (p) => isDfsBook(p.bookmaker, p.prop_source) && p.side.toLowerCase() === "over",
  );

  const picks: EnrichedPick[] = [];

  for (const prop of dfsOverProps) {
    const market = normalizeMarket(prop.market_category);
    if (!market || !SUPPORTED_MARKETS.has(market)) continue;

    const gameId = findGameId(prop, slate.predictions);
    const predKey = prop.player_id && gameId ? `${prop.player_id}:${gameId}` : undefined;
    const preds = predKey ? predIdx.get(predKey) : undefined;
    const model = buildModelFields(market, prop.line, preds, prop.min_roll10);
    const { sharp, consensus } = buildSharpConsensus(prop, market, usProps, oddsPairs);
    const { opp, isHome } = inferOpponent(prop);
    const rate = statRateForMarket(market, prop);

    picks.push({
      player_id: prop.player_id,
      player: prop.player_name,
      display_name: prop.player_name,
      team_abbr: prop.player_team_abbrev,
      opponent_abbr: opp ?? undefined,
      is_home: isHome,
      market,
      dfs_line: prop.line,
      platform: bookToPlatform(prop.bookmaker),
      tier: pickTier(model?.lean, sharp?.lean),
      model,
      sharp,
      consensus,
      game_context: {
        spread: prop.team_spread,
        game_total: prop.game_total,
      },
      form: {},
      vs_opp:
        rate != null && prop.min_roll10
          ? { avg_stat: rate * prop.min_roll10, n_games: 10 }
          : undefined,
    });
  }

  return picks;
}
