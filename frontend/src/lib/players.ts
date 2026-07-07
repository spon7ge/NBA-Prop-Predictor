import type { EnrichedPick, PickEdge, PlayerGroup } from "@/types/slate";
import type { PlayersSortKey } from "@/lib/constants";
import { DFS_BREAK_EVEN, SUPPORTED_MARKETS } from "@/lib/constants";
import { parseSortNumber } from "@/lib/format";

export function pickEdge(pick: EnrichedPick): PickEdge | null {
  const m = pick.model;
  if (!m) return null;
  const pOver = Number(m.p_over);
  if (Number.isNaN(pOver)) return null;
  const pUnder = 1 - pOver;
  if (pOver >= pUnder) {
    return { side: "OVER", prob: pOver, edge: pOver - DFS_BREAK_EVEN };
  }
  return { side: "UNDER", prob: pUnder, edge: pUnder - DFS_BREAK_EVEN };
}

export function edgeBucketClass(edge: number | null): string {
  if (edge == null) return "edge-neg";
  if (edge >= 0.15) return "edge-strong";
  if (edge >= 0.05) return "edge-pos";
  if (edge >= 0) return "edge-marginal";
  return "edge-neg";
}

export function tierPriority(tier: string | undefined): number {
  if (tier === "sharp_verified") return 0;
  if (tier === "conflict") return 1;
  return 2;
}

export function filterPlayerRows(rows: EnrichedPick[], query: string): EnrichedPick[] {
  const q = query.trim().toLowerCase();
  if (!q) return rows.slice();
  const parts = q.split(/\s+/).filter(Boolean);
  return rows.filter((r) => {
    const name = String(r.player || r.display_name || "").toLowerCase();
    return parts.every((part) => name.includes(part));
  });
}

export function filterPlayerRowsByStat(rows: EnrichedPick[], stat: string | null): EnrichedPick[] {
  if (!stat) return rows.slice();
  const want = stat.toUpperCase();
  return rows.filter((r) => String(r.market || "").toUpperCase() === want);
}

export function filterByPlatform(rows: EnrichedPick[], platform: string | null): EnrichedPick[] {
  if (!platform) return rows.slice();
  return rows.filter((r) => String(r.platform || "") === platform);
}

export function filterByTier(rows: EnrichedPick[], tier: string | null): EnrichedPick[] {
  if (!tier) return rows.slice();
  return rows.filter((r) => String(r.tier || "") === tier);
}

export function defaultSortPlayersRows(rows: EnrichedPick[]): EnrichedPick[] {
  return rows.slice().sort((a, b) => {
    const ta = tierPriority(a.tier);
    const tb = tierPriority(b.tier);
    if (ta !== tb) return ta - tb;
    const ma = a.model ? Math.abs(Number(a.model.p_over || 0.5) - 0.5) : 0;
    const mb = b.model ? Math.abs(Number(b.model.p_over || 0.5) - 0.5) : 0;
    if (mb !== ma) return mb - ma;
    return String(a.player || "").localeCompare(String(b.player || ""));
  });
}

function comparePlayersColumn(
  a: EnrichedPick,
  b: EnrichedPick,
  key: PlayersSortKey,
  dir: "asc" | "desc",
): number {
  const strCmp = (get: (r: EnrichedPick) => unknown) => {
    const sa = String(get(a) || "");
    const sb = String(get(b) || "");
    return dir === "desc"
      ? sb.localeCompare(sa, undefined, { sensitivity: "base" })
      : sa.localeCompare(sb, undefined, { sensitivity: "base" });
  };
  const numCmp = (get: (r: EnrichedPick) => unknown) => {
    const va = parseSortNumber(get(a));
    const vb = parseSortNumber(get(b));
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    return dir === "desc" ? vb - va : va - vb;
  };

  switch (key) {
    case "player":
      return strCmp((r) => r.player || r.display_name);
    case "platform":
      return strCmp((r) => r.platform);
    case "mkt":
      return strCmp((r) => r.market);
    case "line":
      return numCmp((r) => r.dfs_line);
    case "tier": {
      const ta = tierPriority(a.tier);
      const tb = tierPriority(b.tier);
      return dir === "desc" ? ta - tb : tb - ta;
    }
    case "modelProb":
      return numCmp((r) => r.model?.p_over);
    case "sharpProb":
      return numCmp((r) => r.sharp?.no_vig_over);
    case "consensusProb":
      return numCmp((r) => r.consensus?.mean_no_vig_over_same_line);
    case "statProj":
      return numCmp((r) => r.model?.stat_q50);
    case "minProj":
      return numCmp((r) => r.model?.min_q50);
    case "l5":
      return numCmp((r) => r.form?.over_l5);
    case "l10":
      return numCmp((r) => r.form?.over_l10);
    case "l15":
      return numCmp((r) => r.form?.over_l15);
    case "vsOppAvg":
      return numCmp((r) => r.vs_opp?.avg_stat);
    case "oppDefRank":
      return numCmp((r) => r.game_context?.opp_def_rating_rank);
    default:
      return 0;
  }
}

export function sortPlayersRows(
  rows: EnrichedPick[],
  sortKey: PlayersSortKey | null,
  sortDir: "asc" | "desc",
): EnrichedPick[] {
  if (!sortKey) return defaultSortPlayersRows(rows);
  return rows.slice().sort((a, b) => comparePlayersColumn(a, b, sortKey, sortDir));
}

export function aggregateEnrichedByPlayer(rows: EnrichedPick[]): PlayerGroup[] {
  const map: Record<string, PlayerGroup> = {};
  const order: string[] = [];

  for (const r of rows) {
    const key = r.player || r.display_name || "";
    if (!map[key]) {
      map[key] = {
        player: key,
        playerId: r.player_id,
        displayName: r.display_name || r.player || "",
        team: r.team_abbr || "",
        opp: r.opponent_abbr || null,
        is_home: r.is_home,
        picks: [],
        tiers: { sharp_verified: 0, conflict: 0, no_model: 0 },
        markets: {},
        platforms: {},
        bestEdge: null,
        bestEdgeSide: null,
      };
      order.push(key);
    }
    const agg = map[key];
    if (r.player_id && !agg.playerId) agg.playerId = r.player_id;
    agg.picks.push(r);
    if (r.market) agg.markets[r.market] = true;
    if (r.platform) agg.platforms[r.platform] = true;
    const tierKey = (r.tier || "no_model") as keyof PlayerGroup["tiers"];
    if (agg.tiers[tierKey] != null) agg.tiers[tierKey] += 1;
    const e = pickEdge(r);
    if (e && (agg.bestEdge == null || e.edge > agg.bestEdge)) {
      agg.bestEdge = e.edge;
      agg.bestEdgeSide = e.side;
    }
  }
  return order.map((k) => map[k]);
}

export function sortPlayerGroups(
  players: PlayerGroup[],
  groupSort: "edge_desc" | "props_desc" | "verified_desc" | "name_asc",
): PlayerGroup[] {
  const n = (v: number | null) => (v == null ? -Infinity : v);
  switch (groupSort) {
    case "props_desc":
      return players.slice().sort((a, b) => b.picks.length - a.picks.length);
    case "verified_desc":
      return players.slice().sort(
        (a, b) =>
          b.tiers.sharp_verified - a.tiers.sharp_verified || b.picks.length - a.picks.length,
      );
    case "name_asc":
      return players.slice().sort((a, b) => String(a.player).localeCompare(String(b.player)));
    case "edge_desc":
    default:
      return players.slice().sort((a, b) => n(b.bestEdge) - n(a.bestEdge));
  }
}

export function filterSupportedPicks(picks: unknown[]): EnrichedPick[] {
  return (picks as EnrichedPick[]).filter((p) => SUPPORTED_MARKETS.has(String(p.market || "")));
}

export function bookPillClass(book: string | undefined): string {
  const b = String(book || "").toLowerCase();
  if (b.includes("prize")) return "book-prizepicks";
  if (b.includes("underdog")) return "book-underdog";
  if (b.includes("draft")) return "book-draftkings";
  if (b.includes("betr")) return "book-betr";
  return "book-unknown";
}

export { comparePlayersColumn };
