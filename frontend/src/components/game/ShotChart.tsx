import { useState } from "react";
import type { GameDetail } from "./types";

const VIEW_WIDTH = 500;
const VIEW_HEIGHT = 470;

function toSvgX(x: number): number {
  return x * 10;
}

function toSvgY(y: number): number {
  return y * 10;
}

type TeamFilter = "both" | string;

export function ShotChart({ detail }: { detail: GameDetail }) {
  const [filter, setFilter] = useState<TeamFilter>("both");

  const visibleShots = detail.shots.filter(
    (shot) => filter === "both" || shot.teamId === filter,
  );

  function teamColor(teamId: string): string {
    if (teamId === detail.away.id) return detail.away.color;
    if (teamId === detail.home.id) return detail.home.color;
    return "#9ca3af";
  }

  const filters: { value: TeamFilter; label: string }[] = [
    { value: "both", label: "Both" },
    { value: detail.away.id, label: detail.away.abbrev },
    { value: detail.home.id, label: detail.home.abbrev },
  ];

  return (
    <div className="rounded-xl border border-white/10 bg-[#141414] p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-white">Shot chart</h2>
        <div className="flex items-center gap-1">
          {filters.map((f) => (
            <button
              key={f.value}
              type="button"
              onClick={() => setFilter(f.value)}
              aria-pressed={filter === f.value}
              className={`rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
                filter === f.value
                  ? "bg-white/15 text-white"
                  : "text-white/50 hover:text-white/80"
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {detail.latestPlay ? (
        <p className="mb-3 truncate rounded-md bg-white/5 px-3 py-2 text-xs text-white/70">
          {detail.latestPlay.text}
        </p>
      ) : (
        <p className="mb-3 text-xs text-white/40">Tip-off pending</p>
      )}

      <svg
        viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
        className="w-full rounded-md bg-[#0d1f16]"
        role="img"
        aria-label="Half-court shot chart"
      >
        <rect
          x={140}
          y={0}
          width={220}
          height={190}
          fill="none"
          stroke="rgba(255,255,255,0.25)"
          strokeWidth={2}
        />
        <circle
          cx={250}
          cy={190}
          r={60}
          fill="none"
          stroke="rgba(255,255,255,0.25)"
          strokeWidth={2}
        />
        <path
          d="M 30 0 L 30 140 A 220 220 0 0 0 470 140 L 470 0"
          fill="none"
          stroke="rgba(255,255,255,0.25)"
          strokeWidth={2}
        />
        <circle
          cx={250}
          cy={40}
          r={7.5}
          fill="none"
          stroke="rgba(255,255,255,0.4)"
          strokeWidth={2}
        />
        {visibleShots.map((shot) => (
          <circle
            key={shot.id}
            role="img"
            aria-label={`${shot.playerName} ${shot.made ? "made" : "missed"} shot`}
            cx={toSvgX(shot.x)}
            cy={toSvgY(shot.y)}
            r={7}
            fill={shot.made ? teamColor(shot.teamId) : "none"}
            stroke={teamColor(shot.teamId)}
            strokeWidth={2}
          />
        ))}
      </svg>

      <div className="mt-3 flex items-center justify-between text-xs text-white/50">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1.5">
            <span className="size-2.5 rounded-full bg-white/70" aria-hidden />
            Made
          </span>
          <span className="flex items-center gap-1.5">
            <span
              className="size-2.5 rounded-full border border-white/70"
              aria-hidden
            />
            Missed
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span>
            {detail.fgMade}/{detail.fgAttempted} FG
          </span>
          <span>Data: ESPN</span>
        </div>
      </div>
    </div>
  );
}
