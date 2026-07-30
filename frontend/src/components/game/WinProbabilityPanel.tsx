import { useEffect, useState, type MouseEvent } from "react";
import type { GameDetail, GameDetailWinProbabilityPoint } from "./types";

const CHART_WIDTH = 320;
const CHART_HEIGHT = 180;

function xForIndex(index: number, count: number, width: number): number {
  if (count <= 1) return width / 2;
  return (index / (count - 1)) * width;
}

function yForPct(pct: number, height: number): number {
  return height - (pct / 100) * height;
}

function nearestIndexForClientX(
  clientX: number,
  rect: DOMRect,
  count: number,
): number {
  if (count <= 1) return 0;
  const ratio = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1);
  return Math.round(ratio * (count - 1));
}

function buildWinProbabilityPath(
  points: GameDetailWinProbabilityPoint[],
  width: number,
  height: number,
): { line: string; area: string } {
  if (points.length === 0) {
    return { line: "", area: "" };
  }

  const coords = points.map((point, index) => ({
    x: xForIndex(index, points.length, width),
    y: yForPct(point.homeWinPct, height),
  }));

  const line = coords
    .map((c, i) => `${i === 0 ? "M" : "L"}${c.x} ${c.y}`)
    .join(" ");

  const midY = height / 2;
  const first = coords[0];
  const last = coords[coords.length - 1];
  const area = [
    `M${first.x} ${midY}`,
    ...coords.map((c) => `L${c.x} ${c.y}`),
    `L${last.x} ${midY}`,
    "Z",
  ].join(" ");

  return { line, area };
}

export function WinProbabilityPanel({ detail }: { detail: GameDetail }) {
  const data = detail.winProbability;
  const points = data?.timeline ?? [];
  const [activeIndex, setActiveIndex] = useState(
    Math.max(points.length - 1, 0),
  );

  useEffect(() => {
    setActiveIndex(Math.max((data?.timeline.length ?? 0) - 1, 0));
  }, [data]);

  if (!data) {
    return (
      <section className="rounded-xl border border-white/10 bg-[#141414] p-4 md:p-5">
        <h2 className="text-sm font-semibold text-white">Win probability</h2>
        <p className="mt-2 text-sm text-white/50">
          Win probability unavailable for this game yet.
        </p>
      </section>
    );
  }

  const activePoint =
    points.length > 0 ? (points[activeIndex] ?? points[points.length - 1]) : null;
  const path = buildWinProbabilityPath(points, CHART_WIDTH, CHART_HEIGHT);

  function handleChartPointerMove(event: MouseEvent<SVGSVGElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    setActiveIndex(nearestIndexForClientX(event.clientX, rect, points.length));
  }

  return (
    <section className="rounded-xl border border-white/10 bg-[#141414] p-4 md:p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-sm font-semibold text-white">Win probability</h2>
          <p className="mt-1 text-xs text-white/50">
            {data.summary ?? `${detail.away.abbrev} vs ${detail.home.abbrev}`}
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <span className="flex items-center gap-1 text-white/60">
            <span
              className="size-2 rounded-full"
              style={{ backgroundColor: detail.away.color }}
            />
            {detail.away.abbrev}
          </span>
          <span className="flex items-center gap-1 text-white/60">
            <span
              className="size-2 rounded-full"
              style={{ backgroundColor: detail.home.color }}
            />
            {detail.home.abbrev}
          </span>
        </div>
      </div>

      {points.length > 0 ? (
        <>
          <svg
            aria-label="Win probability chart"
            viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
            className="mt-4 w-full"
            onMouseMove={handleChartPointerMove}
          >
            <line
              x1={0}
              x2={CHART_WIDTH}
              y1={CHART_HEIGHT / 2}
              y2={CHART_HEIGHT / 2}
              stroke="rgba(255,255,255,0.2)"
              strokeDasharray="4 4"
            />
            <path d={path.area} fill={`${detail.home.color}26`} />
            <path
              d={path.line}
              fill="none"
              stroke={detail.home.color}
              strokeWidth={2.5}
            />
            {activePoint ? (
              <>
                <circle
                  cx={xForIndex(activeIndex, points.length, CHART_WIDTH)}
                  cy={yForPct(activePoint.homeWinPct, CHART_HEIGHT)}
                  r={9}
                  fill={`${detail.home.color}33`}
                  pointerEvents="none"
                />
                <circle
                  cx={xForIndex(activeIndex, points.length, CHART_WIDTH)}
                  cy={yForPct(activePoint.homeWinPct, CHART_HEIGHT)}
                  r={4.5}
                  fill={detail.home.color}
                  pointerEvents="none"
                />
              </>
            ) : null}
          </svg>
          <input
            type="range"
            min={0}
            max={points.length - 1}
            step={1}
            value={activeIndex}
            aria-label="Win probability timeline"
            aria-valuetext={
              activePoint
                ? `Q${activePoint.period} ${activePoint.clock}`
                : undefined
            }
            className="mt-2 w-full accent-orange-500"
            onChange={(event) => {
              setActiveIndex(Number(event.target.value));
            }}
          />
        </>
      ) : null}

      <div className="mt-4 flex items-end justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-wide text-white/45">
            Active state
          </p>
          <div className="mt-1 flex items-baseline gap-2 text-white">
            <span className="font-mono text-lg font-semibold">
              {activePoint
                ? `${activePoint.awayScore}-${activePoint.homeScore}`
                : "—"}
            </span>
            {activePoint ? (
              <span className="text-sm text-white/65">
                Q{activePoint.period} {activePoint.clock}
              </span>
            ) : null}
          </div>
        </div>
        {activePoint ? (
          <div className="flex items-baseline gap-3 text-sm text-white/70">
            <span>
              {detail.home.abbrev} {activePoint.homeWinPct}%
            </span>
            <span>
              {detail.away.abbrev} {activePoint.awayWinPct}%
            </span>
          </div>
        ) : null}
      </div>

      {data.teamStats.length > 0 ? (
        <div className="mt-6 space-y-3">
          {data.teamStats.map((stat) => (
            <div key={stat.key} className="space-y-1.5">
              <div className="grid grid-cols-[40px_1fr_40px] items-center gap-3 text-sm">
                <span className="text-white/80">{stat.awayValue}</span>
                <span className="text-center text-white/55">{stat.label}</span>
                <span className="text-right text-white/80">{stat.homeValue}</span>
              </div>
              <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-2">
                <div
                  className="h-1.5 rounded-full"
                  style={{ backgroundColor: `${detail.away.color}66` }}
                />
                <span className="text-[11px] text-white/35">vs</span>
                <div
                  className="h-1.5 rounded-full"
                  style={{ backgroundColor: `${detail.home.color}66` }}
                />
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}
