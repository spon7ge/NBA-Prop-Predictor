import type { Leg } from "@/types/slate";
import {
  diffDisplay,
  fmt1,
  fmtNumOrDash,
  fmtOrdinalRank,
  hitRate,
  l10StatAvgLabel,
  modelDiffGood,
  ordSuffix,
  pct0,
} from "@/lib/format";

interface LegCardProps {
  leg: Leg;
}

export function LegCard({ leg }: LegCardProps) {
  const isOver = leg.side.toLowerCase() === "over";
  const sideClass = isOver ? "side-over" : "side-under";
  const hr = hitRate(Number(leg.overRateL10 ?? 0), leg.side);
  const hrPct = pct0(hr);
  const diffGood = modelDiffGood(leg.side, leg.prediction, leg.line);
  const diffClass = diffGood ? "diff-pos" : "diff-neg";

  return (
    <div className="leg">
      <p className="player-name">
        {leg.name} <span className="player-team">- {leg.team}</span>
      </p>
      <p className="subtitle">vs {leg.opponent}</p>
      <div className="line-row">
        <span className="line-num">{fmt1(leg.line)}</span>
        <span className="market-lbl">{leg.market}</span>
        <span className={`side-pill ${sideClass}`}>{leg.side.toUpperCase()}</span>
      </div>
      <p className="model-line">
        Model predicts {fmt1(leg.prediction)}{" "}
        <span className={diffClass}>({diffDisplay(leg.side, leg.prediction, leg.line)})</span>
      </p>
      <div className="mini-grid">
        <span>{l10StatAvgLabel(leg.market)}</span>
        <span>{fmtNumOrDash(leg.avgStatL10)}</span>
        <span>vs matchup</span>
        <span>
          {fmtNumOrDash(leg.avgVsMatchup)}
          {leg.matchupGames != null &&
          String(leg.matchupGames).trim() !== "" &&
          !Number.isNaN(Number(leg.matchupGames))
            ? ` (${leg.matchupGames} games)`
            : ""}
        </span>
        <span>Opp Pace Rank</span>
        <span>{fmtOrdinalRank(leg.oppPaceRank)}</span>
        <span>Opp Def Rank</span>
        <span>{ordSuffix(leg.defRank)}</span>
      </div>
      <div className="hit-wrap">
        <div className="hit-label-row">
          <span>Hit rate L10</span>
          <span>{hrPct}%</span>
        </div>
        <div className="hit-track">
          <div className="hit-fill" style={{ width: `${hrPct}%` }} />
        </div>
      </div>
    </div>
  );
}
