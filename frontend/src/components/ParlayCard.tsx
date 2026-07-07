import type { MappedParlay } from "@/types/slate";
import { fmt1, fmtEv, fmtNumOrDash, pct0, spreadFmt } from "@/lib/format";
import { LegCard } from "@/components/LegCard";

interface ParlayCardProps {
  parlay: MappedParlay;
  rank: number;
  nLegs: number;
}

export function ParlayCard({ parlay, rank, nLegs }: ParlayCardProps) {
  const evNum = Number(parlay.ev);
  const evStr = (evNum >= 0 ? "+" : "-") + fmtEv(Math.abs(evNum));
  const probPct = pct0(parlay.parlayProb);
  const kellyPct = fmt1(parlay.kelly ?? 0);
  const cardClass = nLegs > 2 ? `card card--${nLegs}leg` : "card";
  const footerClass = nLegs > 2 ? "card-footer card-footer--multi" : "card-footer";

  return (
    <article className={cardClass}>
      <div className="card-header">
        <span className="rank-label">#{rank} pick</span>
        <div className="badges">
          <span className="pill pill-ev">EV {evStr}%</span>
          <span className="pill pill-prob">Hit prob {probPct}%</span>
          <span className="pill pill-kelly">Kelly {kellyPct}%</span>
        </div>
      </div>
      <div className="legs">
        {parlay.legs.map((leg, i) => (
          <LegCard key={`${leg.name}-${leg.market}-${i}`} leg={leg} />
        ))}
      </div>
      <div className={footerClass}>
        {parlay.legs.map((leg, i) => (
          <div key={`footer-${i}`}>
            Game total {fmtNumOrDash(leg.total)}
            <br />
            Spread {spreadFmt(leg.spread)}
          </div>
        ))}
      </div>
    </article>
  );
}
