import { useEffect, useState, type MouseEvent } from "react";
import type { GameDetail, GameDetailWinProbabilityPoint } from "./types";

const CHART_WIDTH = 640;
const CHART_HEIGHT = 140;
const CHART_PAD_LEFT = 36;
const CHART_PAD_RIGHT = 8;
const CHART_PAD_TOP = 8;
const CHART_PAD_BOTTOM = 6;
const PLOT_WIDTH = CHART_WIDTH - CHART_PAD_LEFT - CHART_PAD_RIGHT;
const PLOT_HEIGHT = CHART_HEIGHT - CHART_PAD_TOP - CHART_PAD_BOTTOM;
const Y_LABEL_X = CHART_PAD_LEFT - 8;

function xForIndex(index: number, count: number): number {
  if (count <= 1) return CHART_PAD_LEFT + PLOT_WIDTH / 2;
  return CHART_PAD_LEFT + (index / (count - 1)) * PLOT_WIDTH;
}

function yForPct(pct: number): number {
  return CHART_PAD_TOP + PLOT_HEIGHT - (pct / 100) * PLOT_HEIGHT;
}

function nearestIndexForClientX(
  clientX: number,
  rect: DOMRect,
  count: number,
): number {
  if (count <= 1) return 0;
  const plotLeft = rect.left + (CHART_PAD_LEFT / CHART_WIDTH) * rect.width;
  const plotWidth = (PLOT_WIDTH / CHART_WIDTH) * rect.width;
  const ratio = Math.min(Math.max((clientX - plotLeft) / plotWidth, 0), 1);
  return Math.round(ratio * (count - 1));
}

function buildWinProbabilityPath(
  points: GameDetailWinProbabilityPoint[],
): { line: string; area: string } {
  if (points.length === 0) {
    return { line: "", area: "" };
  }

  const coords = points.map((point, index) => ({
    x: xForIndex(index, points.length),
    y: yForPct(point.homeWinPct),
  }));

  const line = coords
    .map((c, i) => `${i === 0 ? "M" : "L"}${c.x} ${c.y}`)
    .join(" ");

  const midY = yForPct(50);
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

function SummaryText({
  homeAbbrev,
  homeColor,
}: {
  homeAbbrev: string;
  homeColor: string;
}) {
  return (
    <p className="mt-1 text-xs text-white/50">
      Above the midline favors{" "}
      <span className="font-medium" style={{ color: homeColor }}>
        {homeAbbrev}
      </span>
    </p>
  );
}

function ActiveTooltip({
  detail,
  point,
  index,
  count,
}: {
  detail: GameDetail;
  point: GameDetailWinProbabilityPoint;
  index: number;
  count: number;
}) {
  const markerX = xForIndex(index, count);
  const markerY = yForPct(point.homeWinPct);
  const placeLeft = markerX > CHART_PAD_LEFT + PLOT_WIDTH * 0.75;
  const homeLeads = point.homeWinPct >= point.awayWinPct;

  const leftPct = (markerX / CHART_WIDTH) * 100;
  const topPct = (markerY / CHART_HEIGHT) * 100;

  return (
    <div
      className="pointer-events-none absolute z-10"
      style={{
        left: `${leftPct}%`,
        top: `${topPct}%`,
        transform: placeLeft
          ? "translate(calc(-100% - 12px), -50%)"
          : "translate(12px, -50%)",
      }}
    >
      <div className="whitespace-nowrap rounded-md border border-white/10 bg-black px-2.5 py-1.5">
        <p className="font-mono text-xs font-semibold leading-tight text-white">
          {detail.away.abbrev} {point.awayScore}–{point.homeScore}{" "}
          {detail.home.abbrev}
        </p>
        <p className="mt-0.5 text-[11px] leading-tight">
          <span
            className={homeLeads ? "font-semibold text-white" : "text-white/45"}
          >
            {detail.home.abbrev} {point.homeWinPct}%
          </span>
          <span className="text-white/35"> · </span>
          <span
            className={homeLeads ? "text-white/45" : "font-semibold text-white"}
          >
            {detail.away.abbrev} {point.awayWinPct}%
          </span>
        </p>
      </div>
    </div>
  );
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
      <section>
        <h2 className="text-sm font-semibold text-white">Win probability</h2>
        <p className="mt-2 text-sm text-white/50">
          Win probability unavailable for this game yet.
        </p>
      </section>
    );
  }

  const activePoint =
    points.length > 0 ? (points[activeIndex] ?? points[points.length - 1]) : null;
  const path = buildWinProbabilityPath(points);
  const midY = yForPct(50);
  const gridXs = [0.25, 0.5, 0.75].map(
    (ratio) => CHART_PAD_LEFT + ratio * PLOT_WIDTH,
  );

  function handleChartPointerMove(event: MouseEvent<SVGSVGElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    setActiveIndex(nearestIndexForClientX(event.clientX, rect, points.length));
  }

  return (
    <section>
      <h2 className="text-sm font-semibold text-white">Win probability</h2>
      <SummaryText
        homeAbbrev={detail.home.abbrev}
        homeColor={detail.home.color}
      />

      {points.length > 0 ? (
        <div className="relative mt-4">
          <svg
            aria-label="Win probability chart"
            viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
            className="w-full overflow-visible"
            onMouseMove={handleChartPointerMove}
          >
            <line
              x1={CHART_PAD_LEFT}
              x2={CHART_PAD_LEFT}
              y1={CHART_PAD_TOP}
              y2={CHART_PAD_TOP + PLOT_HEIGHT}
              stroke="#FFFFFF"
              strokeWidth={1}
            />

            <text
              x={Y_LABEL_X}
              y={CHART_PAD_TOP}
              fill="#FFFFFF"
              textAnchor="end"
              dominantBaseline="middle"
              style={{ fontSize: "10px" }}
            >
              100%
            </text>
            <line
              x1={CHART_PAD_LEFT - 5}
              x2={CHART_PAD_LEFT}
              y1={CHART_PAD_TOP}
              y2={CHART_PAD_TOP}
              stroke="#FFFFFF"
              strokeWidth={1}
            />

            <text
              x={Y_LABEL_X}
              y={midY}
              fill="#FFFFFF"
              textAnchor="end"
              dominantBaseline="middle"
              style={{ fontSize: "10px" }}
            >
              50%
            </text>
            <line
              x1={CHART_PAD_LEFT - 5}
              x2={CHART_PAD_LEFT}
              y1={midY}
              y2={midY}
              stroke="#FFFFFF"
              strokeWidth={1}
            />

            {gridXs.map((x) => (
              <line
                key={x}
                x1={x}
                x2={x}
                y1={CHART_PAD_TOP}
                y2={CHART_PAD_TOP + PLOT_HEIGHT}
                stroke="rgba(255,255,255,0.06)"
              />
            ))}

            <line
              x1={CHART_PAD_LEFT}
              x2={CHART_PAD_LEFT + PLOT_WIDTH}
              y1={midY}
              y2={midY}
              stroke="rgba(255,255,255,0.22)"
              strokeDasharray="4 4"
            />

            <path d={path.area} fill={`${detail.home.color}33`} />
            <path
              d={path.line}
              fill="none"
              stroke={detail.home.color}
              strokeWidth={2.25}
              strokeLinejoin="round"
              strokeLinecap="round"
            />

            {activePoint ? (
              <>
                <line
                  x1={xForIndex(activeIndex, points.length)}
                  x2={xForIndex(activeIndex, points.length)}
                  y1={CHART_PAD_TOP}
                  y2={CHART_PAD_TOP + PLOT_HEIGHT}
                  stroke="rgba(255,255,255,0.35)"
                  strokeWidth={1}
                  pointerEvents="none"
                />
                <circle
                  cx={xForIndex(activeIndex, points.length)}
                  cy={yForPct(activePoint.homeWinPct)}
                  r={3.5}
                  fill="#FFFFFF"
                  stroke={detail.home.color}
                  strokeWidth={2}
                  pointerEvents="none"
                />
              </>
            ) : null}
          </svg>

          {activePoint ? (
            <ActiveTooltip
              detail={detail}
              point={activePoint}
              index={activeIndex}
              count={points.length}
            />
          ) : null}

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
            className="sr-only"
            onChange={(event) => {
              setActiveIndex(Number(event.target.value));
            }}
          />
        </div>
      ) : null}

      {data.teamStats.length > 0 ? (
        <div className="mt-6">
          <div className="mb-3 flex items-center justify-between gap-3">
            <h3 className="text-sm font-semibold text-white">Team stats</h3>
            <div className="flex items-center gap-3 text-[10px] text-white/70">
              <span className="flex items-center gap-1.5">
                <span
                  className="size-1.5 rounded-full"
                  style={{ backgroundColor: detail.away.color }}
                />
                {detail.away.abbrev}
              </span>
              <span className="flex items-center gap-1.5">
                <span
                  className="size-1.5 rounded-full"
                  style={{ backgroundColor: detail.home.color }}
                />
                {detail.home.abbrev}
              </span>
            </div>
          </div>

          <div className="space-y-3">
            {data.teamStats.map((stat) => {
              const total = stat.awayValue + stat.homeValue;
              const awayShare =
                total === 0 ? 50 : (stat.awayValue / total) * 100;
              const homeShare =
                total === 0 ? 50 : (stat.homeValue / total) * 100;

              return (
                <div key={stat.key} className="space-y-1.5">
                  <p className="text-center text-[10px] font-medium uppercase tracking-wide text-white">
                    {stat.label}
                  </p>
                  <div className="grid grid-cols-[2rem_1fr_2rem] items-center gap-2">
                    <span className="text-right font-mono text-[10px] text-white">
                      {stat.awayValue}
                    </span>
                    <div className="flex h-1.5 overflow-hidden rounded-sm">
                      <div
                        className="h-full"
                        style={{
                          width: `${awayShare}%`,
                          backgroundColor: detail.away.color,
                        }}
                      />
                      <div
                        className="h-full"
                        style={{
                          width: `${homeShare}%`,
                          backgroundColor: detail.home.color,
                        }}
                      />
                    </div>
                    <span className="font-mono text-[10px] text-white">
                      {stat.homeValue}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ) : null}
    </section>
  );
}
