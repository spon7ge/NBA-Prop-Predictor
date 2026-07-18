import type { Leg } from "@/types/slate";
import {
  diffDisplay,
  fmt1,
  fmtNumOrDash,
  hitRate,
  l10StatAvgLabel,
  modelDiffGood,
  pct0,
} from "@/lib/format";

interface LegCardProps {
  leg: Leg;
}

function hasNum(x: number | string | null | undefined): boolean {
  if (x == null || x === "") return false;
  return !Number.isNaN(Number(x));
}

export function LegCard({ leg }: LegCardProps) {
  const isOver = leg.side.toLowerCase() === "over";
  const sideClass = isOver ? "side-over" : "side-under";
  const hasHitRate = hasNum(leg.overRateL10);
  const hr = hasHitRate ? hitRate(Number(leg.overRateL10), leg.side) : null;
  const hrPct = hr != null ? pct0(hr) : null;
  const diffGood = modelDiffGood(leg.side, leg.prediction, leg.line);
  const diffClass = diffGood ? "diff-pos" : "diff-neg";
  const showL10Avg = hasNum(leg.avgStatL10);

  const opponent =
    leg.opponent != null && String(leg.opponent).trim() !== "" && String(leg.opponent) !== "null"
      ? String(leg.opponent)
      : null;

  return (
    <div className="leg">
      <p className="player-name">
        {leg.name} <span className="player-team">- {leg.team}</span>
      </p>
      {opponent ? <p className="subtitle">vs {opponent}</p> : null}
      <div className="line-row">
        <span className="line-num">{fmt1(leg.line)}</span>
        <span className="market-lbl">{leg.market}</span>
        <span className={`side-pill ${sideClass}`}>{leg.side.toUpperCase()}</span>
      </div>
      <p className="model-line">
        Model predicts {fmt1(leg.prediction)}{" "}
        <span className={diffClass}>({diffDisplay(leg.side, leg.prediction, leg.line)})</span>
      </p>
      {showL10Avg ? (
        <div className="mini-grid">
          <span>{l10StatAvgLabel(leg.market)}</span>
          <span>{fmtNumOrDash(leg.avgStatL10)}</span>
        </div>
      ) : null}
      {hrPct != null ? (
        <div className="hit-wrap">
          <div className="hit-label-row">
            <span>Hit rate L10</span>
            <span>{hrPct}%</span>
          </div>
          <div className="hit-track">
            <div className="hit-fill" style={{ width: `${hrPct}%` }} />
          </div>
        </div>
      ) : null}
    </div>
  );
}
