import type { Book, FlatParlayRow, Leg, LegCount, MappedParlay } from "@/types/slate";
import { BOOKS, SLATE_LEG_COUNTS } from "@/lib/constants";
import { fetchSlateJson, jsonUrls } from "@/lib/api";

const BOOK_FILE_BASE: Record<Book, string> = {
  prizepicks: "prizepicks",
  underdog: "underdog",
  draftkings: "draftKings",
  betr: "betr",
};

export function slateJsonFilename(book: Book, nLegs: LegCount): string {
  const base = BOOK_FILE_BASE[book] || "prizepicks";
  if (nLegs === 2) return `${base}.json`;
  return `${base}_${nLegs}leg.json`;
}

export function normalizeParlayRowForUi(row: FlatParlayRow | null): FlatParlayRow | null {
  if (!row || typeof row !== "object") return null;
  if (row["NAME 1"] != null && row.LEGS == null) return row;
  const legs = row.LEGS;
  if (!Array.isArray(legs) || legs.length < 2) return null;
  if (row.N_LEGS != null && legs.length !== row.N_LEGS) return null;

  const out: FlatParlayRow = {
    PARLAY_PROB: row.PARLAY_PROB,
    EV: row.EV,
    KELLY: row.KELLY != null ? row.KELLY : row.KELLY_QUARTER,
  };

  for (let i = 0; i < legs.length; i += 1) {
    const L = legs[i];
    const n = i + 1;
    const gc = L.game_context || {};
    const vs = L.vs_opp || {};
    const fo = L.form || {};
    const md = L.model || {};
    out[`NAME ${n}`] = L.player != null ? L.player : L.display_name;
    out[`TEAM ${n}`] = L.team_abbr != null ? L.team_abbr : L.team;
    out[`MARKET ${n}`] = L.market;
    out[`LINE ${n}`] = L.dfs_line;
    out[`SIDE ${n}`] = L.side != null ? L.side : md.lean;
    out[`PREDICTION ${n}`] = md.stat_q50;
    out[`OPPONENT ${n}`] =
      L.opponent_abbr != null && String(L.opponent_abbr).trim() !== ""
        ? L.opponent_abbr
        : L.opponent;
    out[`SPREAD ${n}`] = gc.spread;
    out[`TOTAL ${n}`] = gc.game_total;
    out[`OPP_DEF_RANK ${n}`] = gc.opp_def_rating_rank;
    out[`OPP_PACE_RANK ${n}`] = gc.opp_pace_rank;
    out[`AVG_STAT_L10 ${n}`] = L.avg_stat_l10;
    out[`AVG_STAT_VS_MATCHUP ${n}`] = vs.avg_stat;
    out[`MATCHUP_GAMES ${n}`] = vs.n_games;
    out[`OVER_RATE_L10 ${n}`] = fo.over_l10;
  }
  return out;
}

export function normalizeSlateArray(arr: FlatParlayRow[]): FlatParlayRow[] {
  const out: FlatParlayRow[] = [];
  for (const row of arr) {
    const normalized = normalizeParlayRowForUi(row);
    if (normalized) out.push(normalized);
  }
  return out;
}

function sortByEvDesc(arr: FlatParlayRow[]): FlatParlayRow[] {
  return arr.slice().sort((a, b) => Number(b.EV) - Number(a.EV));
}

export function legFromRow(row: FlatParlayRow, i: number): Leg {
  return {
    name: String(row[`NAME ${i}`] ?? ""),
    team: String(row[`TEAM ${i}`] ?? ""),
    market: String(row[`MARKET ${i}`] ?? ""),
    line: Number(row[`LINE ${i}`]),
    side: String(row[`SIDE ${i}`] ?? ""),
    prediction: Number(row[`PREDICTION ${i}`]),
    opponent: String(row[`OPPONENT ${i}`] ?? ""),
    spread: row[`SPREAD ${i}`] as number | undefined,
    total: row[`TOTAL ${i}`] as number | undefined,
    defRank: row[`OPP_DEF_RANK ${i}`] as number | undefined,
    avgStatL10: row[`AVG_STAT_L10 ${i}`] as number | undefined,
    oppPaceRank: row[`OPP_PACE_RANK ${i}`] as number | undefined,
    avgVsMatchup: row[`AVG_STAT_VS_MATCHUP ${i}`] as number | undefined,
    matchupGames: row[`MATCHUP_GAMES ${i}`] as number | undefined,
    overRateL10: row[`OVER_RATE_L10 ${i}`] as number | undefined,
  };
}

export function mapRowN(row: FlatParlayRow, nLegs: number): MappedParlay {
  const legs: Leg[] = [];
  for (let i = 1; i <= nLegs; i += 1) {
    legs.push(legFromRow(row, i));
  }
  return {
    parlayProb: Number(row.PARLAY_PROB),
    ev: Number(row.EV),
    kelly: row.KELLY != null ? Number(row.KELLY) : undefined,
    legs,
  };
}

export async function loadAllSlates(): Promise<Record<LegCount, Record<Book, FlatParlayRow[]>>> {
  const fetches: Promise<FlatParlayRow[]>[] = [];
  for (const nLegs of SLATE_LEG_COUNTS) {
    for (const book of BOOKS) {
      fetches.push(fetchSlateJson<FlatParlayRow>(jsonUrls(slateJsonFilename(book, nLegs))));
    }
  }

  const results = await Promise.all(fetches);
  const slates = {} as Record<LegCount, Record<Book, FlatParlayRow[]>>;
  let idx = 0;
  for (const nLegs of SLATE_LEG_COUNTS) {
    slates[nLegs] = {} as Record<Book, FlatParlayRow[]>;
    for (const book of BOOKS) {
      slates[nLegs][book] = sortByEvDesc(normalizeSlateArray(results[idx] ?? []));
      idx += 1;
    }
  }
  return slates;
}

export function hasAnySlates(
  slates: Record<LegCount, Record<Book, FlatParlayRow[]>>,
): boolean {
  for (const nLegs of SLATE_LEG_COUNTS) {
    const bucket = slates[nLegs];
    for (const book of BOOKS) {
      if ((bucket[book] ?? []).length > 0) return true;
    }
  }
  return false;
}
