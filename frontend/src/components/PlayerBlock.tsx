import type { ReactNode } from "react";
import type { EnrichedPick, PlayerGroup } from "@/types/slate";
import type { PlayersSortKey } from "@/lib/constants";
import {
  bookPillClass,
  comparePlayersColumn,
  edgeBucketClass,
  pickEdge,
} from "@/lib/players";
import { PlayerStatsPanel } from "@/components/PlayerStatsPanel";
import {
  fmt1,
  fmtNumOrDash,
  fmtOrdinalRank,
  fmtOverRatePct,
} from "@/lib/format";

function BookPill({ book }: { book?: string }) {
  const label = String(book || "").trim() || "—";
  return <span className={`book-pill ${bookPillClass(book)}`}>{label}</span>;
}

function EdgeCell({ pick }: { pick: EnrichedPick }) {
  const e = pickEdge(pick);
  if (!e) return <span className="edge-cell edge-neg">—</span>;
  const sign = e.edge >= 0 ? "+" : "";
  return (
    <span className={`edge-cell ${edgeBucketClass(e.edge)}`}>
      {sign}
      {(e.edge * 100).toFixed(1)}%
    </span>
  );
}

function BestSidePill({ pick }: { pick: EnrichedPick }) {
  const e = pickEdge(pick);
  if (!e) return null;
  const cls = e.side === "OVER" ? "side-over" : "side-under";
  return <span className={`side-pill ${cls}`}>{e.side}</span>;
}

function ModelHitProbs({
  pOver,
  pUnder,
}: {
  pOver?: number | null;
  pUnder?: number | null;
}) {
  if (pOver == null && pUnder == null) return <>—</>;
  const under =
    pUnder != null && !Number.isNaN(Number(pUnder))
      ? Number(pUnder)
      : pOver != null && !Number.isNaN(Number(pOver))
        ? 1 - Number(pOver)
        : null;
  return (
    <span className="model-hit-probs">
      <span className="model-hit-probs__over">Over {fmtOverRatePct(pOver)}</span>
      <span className="model-hit-probs__sep" aria-hidden="true">
        ·
      </span>
      <span className="model-hit-probs__under">Under {fmtOverRatePct(under)}</span>
    </span>
  );
}

interface SortThProps {
  sortKey: PlayersSortKey;
  label: string;
  activeKey: PlayersSortKey | null;
  sortDir: "asc" | "desc";
  onSort: (key: PlayersSortKey) => void;
  className?: string;
  title?: string;
}

function SortTh({ sortKey, label, activeKey, sortDir, onSort, className = "", title }: SortThProps) {
  const active = activeKey === sortKey;
  const thClass =
    `players-sort-th${className ? ` ${className}` : ""}${active ? " players-sort-th--active" : ""}`;
  const ariaSort = active ? (sortDir === "desc" ? "descending" : "ascending") : "none";
  const arrow = active ? (sortDir === "desc" ? " ↓" : " ↑") : "";
  return (
    <th className={thClass} title={title} aria-sort={ariaSort as "none" | "ascending" | "descending"}>
      <button type="button" className="players-sort-btn" onClick={() => onSort(sortKey)}>
        {label}
        {arrow}
      </button>
    </th>
  );
}

interface PlayerExpandedBodyProps {
  picks: EnrichedPick[];
  sortKey: PlayersSortKey | null;
  sortDir: "asc" | "desc";
  onSort: (key: PlayersSortKey) => void;
}

function PlayerExpandedBody({ picks, sortKey, sortDir, onSort }: PlayerExpandedBodyProps) {
  const sorted = picks.slice().sort((a, b) => {
    if (sortKey) return comparePlayersColumn(a, b, sortKey, sortDir);
    const ea = pickEdge(a);
    const eb = pickEdge(b);
    const va = ea ? ea.edge : -Infinity;
    const vb = eb ? eb.edge : -Infinity;
    return vb - va;
  });

  return (
    <>
      <div className="players-wrap">
        <table className="players-table players-table--sortable">
          <thead>
            <tr>
              <SortTh sortKey="platform" label="Book" activeKey={sortKey} sortDir={sortDir} onSort={onSort} />
              <SortTh sortKey="mkt" label="Market" activeKey={sortKey} sortDir={sortDir} onSort={onSort} />
              <SortTh sortKey="line" label="Line" activeKey={sortKey} sortDir={sortDir} onSort={onSort} className="num" />
              <th className="num">Edge</th>
              <SortTh sortKey="modelProb" label="Over / Under" activeKey={sortKey} sortDir={sortDir} onSort={onSort} className="col-over-under" />
              <SortTh sortKey="statProj" label="Proj" activeKey={sortKey} sortDir={sortDir} onSort={onSort} className="num col-secondary" />
              <SortTh sortKey="minProj" label="Min" activeKey={sortKey} sortDir={sortDir} onSort={onSort} className="num col-secondary" />
              <SortTh sortKey="l5" label="L5" activeKey={sortKey} sortDir={sortDir} onSort={onSort} className="num" />
              <SortTh sortKey="l10" label="L10" activeKey={sortKey} sortDir={sortDir} onSort={onSort} className="num" />
              <SortTh sortKey="l15" label="L15" activeKey={sortKey} sortDir={sortDir} onSort={onSort} className="num col-secondary" />
              <SortTh sortKey="vsOppAvg" label="vs Opp" activeKey={sortKey} sortDir={sortDir} onSort={onSort} className="num col-secondary" />
              <SortTh sortKey="oppDefRank" label="Def Rnk" activeKey={sortKey} sortDir={sortDir} onSort={onSort} className="num col-secondary" />
            </tr>
          </thead>
          <tbody>
            {sorted.map((r, i) => {
              const model = r.model || {};
              const gc = r.game_context || {};
              const form = r.form || {};
              const vsOpp = r.vs_opp || {};
              return (
                <tr key={i}>
                  <td>
                    <BookPill book={r.platform} />
                  </td>
                  <td>{r.market || ""}</td>
                  <td className="num">{fmt1(r.dfs_line)}</td>
                  <td className="num">
                    <EdgeCell pick={r} />
                  </td>
                  <td className="enriched-lean col-over-under">
                    <ModelHitProbs pOver={model.p_over} pUnder={model.p_under} />
                  </td>
                  <td className="num col-secondary">{fmtNumOrDash(model.stat_q50)}</td>
                  <td className="num col-secondary">{fmtNumOrDash(model.min_q50)}</td>
                  <td className="num">{fmtOverRatePct(form.over_l5)}</td>
                  <td className="num">{fmtOverRatePct(form.over_l10)}</td>
                  <td className="num col-secondary">{fmtOverRatePct(form.over_l15)}</td>
                  <td className="num enriched-vsopp col-secondary">
                    {vsOpp.avg_stat != null ? (
                      <>
                        {fmt1(vsOpp.avg_stat)}{" "}
                        <span className="enriched-dim">({vsOpp.n_games || 0}g)</span>
                      </>
                    ) : (
                      "—"
                    )}
                  </td>
                  <td className="num col-secondary">{fmtOrdinalRank(gc.opp_def_rating_rank)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="props-cards" aria-hidden="true">
        {sorted.map((p, j) => {
          const m = p.model || {};
          const f = p.form || {};
          return (
            <div className="props-cards-prop" key={j}>
              <div className="props-cards-prop-header">
                <span className="props-cards-prop-mkt">{p.market}</span>
                <span className="props-cards-prop-line">{fmt1(p.dfs_line)}</span>
                <BestSidePill pick={p} />
                <BookPill book={p.platform} />
              </div>
              <div className="props-cards-prop-meta">
                <span>
                  Edge <EdgeCell pick={p} />
                </span>
                <span>
                  <ModelHitProbs pOver={m.p_over} pUnder={m.p_under} />
                </span>
                <span>
                  L5/10: {fmtOverRatePct(f.over_l5)} · {fmtOverRatePct(f.over_l10)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}

interface PlayerBlockProps {
  player: PlayerGroup;
  expanded: boolean;
  onToggle: () => void;
  sortKey: PlayersSortKey | null;
  sortDir: "asc" | "desc";
  onSort: (key: PlayersSortKey) => void;
}

export function PlayerBlock({
  player: p,
  expanded,
  onToggle,
  sortKey,
  sortDir,
  onSort,
}: PlayerBlockProps) {
  const totalProps = p.picks.length;
  const nMarkets = Object.keys(p.markets).length;
  const nPlatforms = Object.keys(p.platforms).length;
  const matchupLabel = p.opp ? `${p.is_home ? "vs " : "@ "}${p.opp}` : "no opp";

  let edgeHtml: ReactNode;
  if (p.bestEdge == null) {
    edgeHtml = <span className="edge-cell edge-neg">—</span>;
  } else {
    const sign = p.bestEdge >= 0 ? "+" : "";
    edgeHtml = (
      <span className={`edge-cell ${edgeBucketClass(p.bestEdge)}`}>
        {sign}
        {(p.bestEdge * 100).toFixed(1)}%
      </span>
    );
  }

  const sideHtml = p.bestEdgeSide ? (
    <span className={`side-pill side-${p.bestEdgeSide.toLowerCase()}`}>{p.bestEdgeSide}</span>
  ) : null;

  return (
    <div className={`player-block${expanded ? " player-block--expanded" : ""}`} data-player={p.player}>
      <button
        type="button"
        className="player-row"
        aria-expanded={expanded}
        onClick={onToggle}
      >
        <div className="player-row-main">
          <span className="player-row-name">{p.displayName}</span>
          <span className="player-row-team">
            {p.team} · {matchupLabel}
          </span>
        </div>
        <div className="player-row-stats">
          <span className="player-row-stat">
            <b>{totalProps}</b> {totalProps === 1 ? "prop" : "props"} ·{" "}
            <span className="dim">
              {nMarkets} {nMarkets === 1 ? "market" : "markets"} ·{" "}
              {nPlatforms} {nPlatforms === 1 ? "book" : "books"}
            </span>
          </span>
          <span className="player-row-best">
            {edgeHtml} {sideHtml}
          </span>
        </div>
        <span className="player-row-chevron" aria-hidden="true" />
      </button>
      {expanded && (
        <div className="player-block-body">
          <PlayerStatsPanel playerId={p.playerId} enabled={expanded} />
          <PlayerExpandedBody picks={p.picks} sortKey={sortKey} sortDir={sortDir} onSort={onSort} />
        </div>
      )}
    </div>
  );
}
