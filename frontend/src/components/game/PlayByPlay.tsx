import { useMemo, useState } from "react";
import type { GameDetail } from "./types";

function periodLabel(period: number): string {
  const ordinals = ["1st", "2nd", "3rd", "4th"];
  if (period <= ordinals.length) return ordinals[period - 1];
  const ot = period - ordinals.length;
  return ot === 1 ? "OT" : `${ot}OT`;
}

export function PlayByPlay({ detail }: { detail: GameDetail }) {
  const periods = useMemo(() => {
    const set = new Set(detail.plays.map((play) => play.period));
    return Array.from(set).sort((a, b) => a - b);
  }, [detail.plays]);

  const [selectedPeriod, setSelectedPeriod] = useState<number | null>(null);
  const activePeriod =
    selectedPeriod ?? (periods.length > 0 ? periods[periods.length - 1] : null);

  function teamColor(teamId: string | null): string {
    if (teamId === detail.away.id) return detail.away.color;
    if (teamId === detail.home.id) return detail.home.color;
    return "#9ca3af";
  }

  // `detail.plays` is already newest-first (matches the API response order).
  // Cap the list at 10 so the panel stays scannable during long quarters.
  const playsForPeriod = detail.plays
    .filter((play) => play.period === activePeriod)
    .slice(0, 10);

  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-[14px] font-semibold text-white">Play-by-play</h2>
        <div className="flex items-center gap-1">
          {periods.map((period) => (
            <button
              key={period}
              type="button"
              onClick={() => setSelectedPeriod(period)}
              aria-pressed={activePeriod === period}
              className={`rounded-full px-2.5 py-1 text-[14px] font-medium transition-colors ${
                activePeriod === period
                  ? "bg-white/15 text-white"
                  : "text-white/50 hover:text-white/80"
              }`}
            >
              {periodLabel(period)}
            </button>
          ))}
        </div>
      </div>

      {playsForPeriod.length === 0 ? (
        <p className="text-[14px] text-white/40">Tip-off pending</p>
      ) : (
        <ul className="space-y-1.5 text-[14px]">
          {playsForPeriod.map((play, index) => (
            <li
              key={play.id}
              className={`flex items-center gap-2 rounded-md px-2 py-1.5 ${
                play.scoring ? "bg-white/5" : ""
              } ${
                index === 0
                  ? "bg-white/10 font-semibold text-white ring-1 ring-white/25"
                  : ""
              }`}
            >
              <span
                className="size-2 shrink-0 rounded-full"
                style={{ backgroundColor: teamColor(play.teamId) }}
                aria-hidden
              />
              <span
                className={`w-10 shrink-0 font-mono ${
                  index === 0 ? "text-white/60" : "text-white/40"
                }`}
              >
                {play.clock}
              </span>
              <span
                className={`min-w-0 flex-1 ${
                  index === 0 ? "text-white" : "text-white/80"
                }`}
              >
                {play.text}
              </span>
              {play.scoring ? (
                <span className="shrink-0 font-mono font-semibold text-amber-300">
                  {play.awayScore}-{play.homeScore}
                </span>
              ) : null}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
