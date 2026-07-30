import { useMemo, useState } from "react";
import type { GameDetail } from "./types";

/** Half-court in feet × 10 (ESPN: x 0–50, y 0–47 with y≈0 at the basket). */
const VIEW_WIDTH = 500;
const VIEW_HEIGHT = 470;
const LINE = "rgba(255,255,255,0.35)";
const LINE_WIDTH = 1.5;

function toSvgX(x: number): number {
  return x * 10;
}

/** Flip Y so the basket sits at the bottom of the chart (matches the mockup). */
function toSvgY(y: number): number {
  return VIEW_HEIGHT - y * 10;
}

type TeamFilter = "both" | string;
type PeriodFilter = "all" | number;

function periodClockLabel(period: number, clock: string): string {
  return `Q${period} ${clock}`;
}

function quarterLabel(period: number): string {
  if (period <= 4) return `Q${period}`;
  const ot = period - 4;
  return ot === 1 ? "OT" : `${ot}OT`;
}

export function ShotChart({ detail }: { detail: GameDetail }) {
  const [teamFilter, setTeamFilter] = useState<TeamFilter>("both");
  const [periodFilter, setPeriodFilter] = useState<PeriodFilter>("all");

  const periods = useMemo(() => {
    const set = new Set(detail.shots.map((shot) => shot.period));
    for (const play of detail.plays) set.add(play.period);
    return Array.from(set)
      .filter((p) => p > 0)
      .sort((a, b) => a - b);
  }, [detail.shots, detail.plays]);

  const visibleShots = detail.shots.filter((shot) => {
    if (teamFilter !== "both" && shot.teamId !== teamFilter) return false;
    if (periodFilter !== "all" && shot.period !== periodFilter) return false;
    return true;
  });

  const fgMade = visibleShots.filter((s) => s.made).length;
  const fgAttempted = visibleShots.length;

  function teamColor(teamId: string): string {
    if (teamId === detail.away.id) return detail.away.color;
    if (teamId === detail.home.id) return detail.home.color;
    return "#9ca3af";
  }

  const highlightedShotId = (() => {
    const latest = detail.latestPlay;
    if (!latest) return null;
    const byId = visibleShots.find((s) => s.id === latest.id);
    if (byId) return byId.id;
    const byClock = visibleShots.find(
      (s) =>
        s.period === latest.period &&
        s.clock === latest.clock &&
        (!latest.teamId || s.teamId === latest.teamId),
    );
    return byClock?.id ?? null;
  })();

  const teamFilters: { value: TeamFilter; label: string }[] = [
    { value: "both", label: "Both" },
    { value: detail.away.id, label: detail.away.abbrev },
    { value: detail.home.id, label: detail.home.abbrev },
  ];

  const periodFilters: { value: PeriodFilter; label: string }[] = [
    { value: "all", label: "All" },
    ...periods.map((period) => ({
      value: period as PeriodFilter,
      label: quarterLabel(period),
    })),
  ];

  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-white">Shot chart</h2>
        <div className="flex items-center gap-1">
          {teamFilters.map((f) => (
            <button
              key={f.value}
              type="button"
              onClick={() => setTeamFilter(f.value)}
              aria-pressed={teamFilter === f.value}
              className={`rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
                teamFilter === f.value
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
        <div className="mb-3 flex items-center gap-2.5 rounded-xl border border-white/15 px-3 py-2.5">
          <span
            className="size-2 shrink-0 rounded-full bg-emerald-500"
            aria-hidden
          />
          <p className="min-w-0 flex-1 truncate text-[14px] text-white/90">
            {detail.latestPlay.text}
          </p>
          <span className="shrink-0 text-[14px] text-white/70">
            {periodClockLabel(detail.latestPlay.period, detail.latestPlay.clock)}
          </span>
        </div>
      ) : (
        <p className="mb-3 text-[14px] text-white/40">Tip-off pending</p>
      )}

      <svg
        viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
        className="w-full"
        role="img"
        aria-label="Half-court shot chart"
      >
        {/* Outer boundary — baseline at bottom */}
        <rect
          x={0}
          y={0}
          width={VIEW_WIDTH}
          height={VIEW_HEIGHT}
          fill="none"
          stroke={LINE}
          strokeWidth={LINE_WIDTH}
        />

        {/* Paint / key (16 ft wide × 19 ft deep from baseline) */}
        <rect
          x={170}
          y={VIEW_HEIGHT - 190}
          width={160}
          height={190}
          fill="none"
          stroke={LINE}
          strokeWidth={LINE_WIDTH}
        />

        {/* Free-throw circle — upper half only (top of the key) */}
        <path
          d={`M 190 ${VIEW_HEIGHT - 190} A 60 60 0 0 0 310 ${VIEW_HEIGHT - 190}`}
          fill="none"
          stroke={LINE}
          strokeWidth={LINE_WIDTH}
        />
        <line
          x1={170}
          y1={VIEW_HEIGHT - 190}
          x2={330}
          y2={VIEW_HEIGHT - 190}
          stroke={LINE}
          strokeWidth={LINE_WIDTH}
        />

        {/* Restricted-area arc */}
        <path
          d={`M 210 ${VIEW_HEIGHT - 52.5} A 40 40 0 0 1 290 ${VIEW_HEIGHT - 52.5}`}
          fill="none"
          stroke={LINE}
          strokeWidth={LINE_WIDTH}
        />

        {/* Backboard */}
        <line
          x1={220}
          y1={VIEW_HEIGHT - 40}
          x2={280}
          y2={VIEW_HEIGHT - 40}
          stroke={LINE}
          strokeWidth={2}
        />

        {/* Rim */}
        <circle
          cx={250}
          cy={VIEW_HEIGHT - 52.5}
          r={7.5}
          fill="none"
          stroke={LINE}
          strokeWidth={LINE_WIDTH}
        />

        {/* Three-point line — WNBA-ish corners + arc (basket at bottom) */}
        <path
          d={`M 33 ${VIEW_HEIGHT}
              L 33 ${VIEW_HEIGHT - 140}
              A 237 237 0 0 1 467 ${VIEW_HEIGHT - 140}
              L 467 ${VIEW_HEIGHT}`}
          fill="none"
          stroke={LINE}
          strokeWidth={LINE_WIDTH}
        />

        {visibleShots.map((shot) => {
          const color = teamColor(shot.teamId);
          const cx = toSvgX(shot.x);
          const cy = toSvgY(shot.y);
          const isHighlight = shot.id === highlightedShotId;
          return (
            <g key={shot.id}>
              {isHighlight ? (
                <circle
                  cx={cx}
                  cy={cy}
                  r={14}
                  fill="none"
                  stroke={color}
                  strokeWidth={1.5}
                  opacity={0.45}
                  aria-hidden
                />
              ) : null}
              <circle
                role="img"
                aria-label={`${shot.playerName} ${shot.made ? "made" : "missed"} shot`}
                cx={cx}
                cy={cy}
                r={6}
                fill={shot.made ? color : "none"}
                stroke={color}
                strokeWidth={1.75}
              />
            </g>
          );
        })}
      </svg>

      <div className="mt-3 mb-3 flex items-center gap-1">
        {periodFilters.map((f) => (
          <button
            key={String(f.value)}
            type="button"
            onClick={() => setPeriodFilter(f.value)}
            aria-pressed={periodFilter === f.value}
            className={`rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
              periodFilter === f.value
                ? "bg-white/15 text-white"
                : "text-white/50 hover:text-white/80"
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      <div className="flex items-center justify-between text-xs text-white/50">
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
            {fgMade}/{fgAttempted} FG
          </span>
          <span aria-hidden>·</span>
          <span>Data: ESPN</span>
        </div>
      </div>
    </div>
  );
}
