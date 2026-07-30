import type { GameDetail } from "./types";

type MatchupPredictionProps = {
  detail: GameDetail;
};

export function MatchupPrediction({ detail }: MatchupPredictionProps) {
  const prediction = detail.matchupPrediction;

  if (!prediction) {
    return null;
  }

  return (
    <section className="rounded-xl border border-white/10 bg-[#141414] p-4">
      <h2 className="text-sm font-semibold text-white">Matchup prediction</h2>

      <div className="mt-3 flex h-2 overflow-hidden rounded-full">
        <div
          className="h-full"
          style={{
            width: `${prediction.awayWinPct}%`,
            backgroundColor: detail.away.color,
          }}
        />
        <div
          className="h-full"
          style={{
            width: `${prediction.homeWinPct}%`,
            backgroundColor: detail.home.color,
          }}
        />
      </div>

      <div className="mt-2 flex items-center justify-between text-xs text-white/70">
        <span>
          {detail.away.abbrev}{" "}
          <span>{`${prediction.awayWinPct}%`}</span>
        </span>
        <span>
          <span>{`${prediction.homeWinPct}%`}</span> {detail.home.abbrev}
        </span>
      </div>

      <p className="mt-2 text-xs text-white/50">{prediction.sourceLabel}</p>
    </section>
  );
}
