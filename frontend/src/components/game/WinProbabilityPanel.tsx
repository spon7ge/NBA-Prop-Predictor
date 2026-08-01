import { useEffect, useState, type MouseEvent } from "react";
import { GameSection } from "./GameSection";
import type { GameDetail } from "./types";
import {
  buildSplitSeriesPaths,
  CHART_GEOMETRY,
  nearestIndexForClientX,
  xForIndex,
  yForPct,
} from "./winProbabilityPaths";

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
      <GameSection>
        <h2 className="text-sm font-semibold text-white">Win probability</h2>
        <p className="mt-2 text-sm text-white/50">
          Win probability unavailable for this game yet.
        </p>
      </GameSection>
    );
  }

  const scrub = Math.min(
    Math.max(activeIndex, 0),
    Math.max(points.length - 1, 0),
  );
  const paths = buildSplitSeriesPaths(points, scrub);
  const activePoint =
    points.length > 0 ? (points[scrub] ?? points[points.length - 1]) : null;
  const midY = yForPct(50);
  const gridXs = [0.25, 0.5, 0.75].map(
    (ratio) => CHART_GEOMETRY.padLeft + ratio * CHART_GEOMETRY.plotWidth,
  );

  const vividProps = {
    fill: "none" as const,
    strokeWidth: 2.25,
    strokeLinejoin: "round" as const,
    strokeLinecap: "round" as const,
    "data-wp-segment": "vivid",
  };
  const mutedProps = {
    fill: "none" as const,
    strokeWidth: 2.25,
    strokeLinejoin: "round" as const,
    strokeLinecap: "round" as const,
    stroke: "rgba(255,255,255,0.28)",
    opacity: 0.35,
    "data-wp-segment": "muted",
  };

  function handleChartPointerMove(event: MouseEvent<SVGSVGElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    setActiveIndex(nearestIndexForClientX(event.clientX, rect, points.length));
  }

  const scrubX = xForIndex(scrub, points.length);
  const placeLeft =
    scrubX >
    CHART_GEOMETRY.padLeft + CHART_GEOMETRY.plotWidth * 0.75;
  const labelX = placeLeft ? scrubX - 8 : scrubX + 8;
  const labelAnchor = placeLeft ? "end" : "start";

  return (
    <GameSection>
      <h2 className="text-sm font-semibold text-white">Win probability</h2>

      {points.length > 0 ? (
        <div className="relative mt-4">
          <svg
            aria-label="Win probability chart"
            viewBox={`0 0 ${CHART_GEOMETRY.width} ${CHART_GEOMETRY.height}`}
            className="w-full overflow-visible"
            onMouseMove={handleChartPointerMove}
          >
            <line
              x1={CHART_GEOMETRY.padLeft}
              x2={CHART_GEOMETRY.padLeft}
              y1={CHART_GEOMETRY.padTop}
              y2={CHART_GEOMETRY.padTop + CHART_GEOMETRY.plotHeight}
              stroke="#FFFFFF"
              strokeWidth={1}
            />

            <text
              x={CHART_GEOMETRY.yLabelX}
              y={CHART_GEOMETRY.padTop}
              fill="#FFFFFF"
              textAnchor="end"
              dominantBaseline="middle"
              style={{ fontSize: "10px" }}
            >
              100%
            </text>
            <line
              x1={CHART_GEOMETRY.padLeft - 5}
              x2={CHART_GEOMETRY.padLeft}
              y1={CHART_GEOMETRY.padTop}
              y2={CHART_GEOMETRY.padTop}
              stroke="#FFFFFF"
              strokeWidth={1}
            />

            <text
              x={CHART_GEOMETRY.yLabelX}
              y={midY}
              fill="#FFFFFF"
              textAnchor="end"
              dominantBaseline="middle"
              style={{ fontSize: "10px" }}
            >
              50%
            </text>
            <line
              x1={CHART_GEOMETRY.padLeft - 5}
              x2={CHART_GEOMETRY.padLeft}
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
                y1={CHART_GEOMETRY.padTop}
                y2={CHART_GEOMETRY.padTop + CHART_GEOMETRY.plotHeight}
                stroke="rgba(255,255,255,0.06)"
              />
            ))}

            <line
              x1={CHART_GEOMETRY.padLeft}
              x2={CHART_GEOMETRY.padLeft + CHART_GEOMETRY.plotWidth}
              y1={midY}
              y2={midY}
              stroke="rgba(255,255,255,0.22)"
              strokeDasharray="4 4"
            />

            {paths.awayVivid ? (
              <path
                d={paths.awayVivid}
                stroke={detail.away.color}
                {...vividProps}
              />
            ) : null}
            {paths.homeVivid ? (
              <path
                d={paths.homeVivid}
                stroke={detail.home.color}
                {...vividProps}
              />
            ) : null}
            {paths.awayMuted ? (
              <path d={paths.awayMuted} {...mutedProps} />
            ) : null}
            {paths.homeMuted ? (
              <path d={paths.homeMuted} {...mutedProps} />
            ) : null}

            {activePoint ? (
              <>
                <line
                  x1={scrubX}
                  x2={scrubX}
                  y1={CHART_GEOMETRY.padTop}
                  y2={CHART_GEOMETRY.padTop + CHART_GEOMETRY.plotHeight}
                  stroke="rgba(255,255,255,0.45)"
                  strokeDasharray="3 3"
                  pointerEvents="none"
                />
                <circle
                  cx={scrubX}
                  cy={yForPct(activePoint.awayWinPct)}
                  r={3.5}
                  fill={detail.away.color}
                  stroke="#FFFFFF"
                  strokeWidth={1.5}
                  pointerEvents="none"
                />
                <circle
                  cx={scrubX}
                  cy={yForPct(activePoint.homeWinPct)}
                  r={3.5}
                  fill={detail.home.color}
                  stroke="#FFFFFF"
                  strokeWidth={1.5}
                  pointerEvents="none"
                />
                <text
                  x={labelX}
                  y={yForPct(activePoint.homeWinPct)}
                  fill={detail.home.color}
                  textAnchor={labelAnchor}
                  dominantBaseline="middle"
                  style={{ fontSize: "11px", fontWeight: 600 }}
                >
                  {detail.home.abbrev} {activePoint.homeWinPct}%
                </text>
                <text
                  x={labelX}
                  y={yForPct(activePoint.awayWinPct)}
                  fill={detail.away.color}
                  textAnchor={labelAnchor}
                  dominantBaseline="middle"
                  style={{ fontSize: "11px", fontWeight: 600 }}
                >
                  {detail.away.abbrev} {activePoint.awayWinPct}%
                </text>
                <text
                  x={scrubX}
                  y={CHART_GEOMETRY.padTop + 10}
                  fill="rgba(255,255,255,0.7)"
                  textAnchor="middle"
                  data-wp-clock
                  style={{ fontSize: "10px" }}
                >
                  {`Q${activePoint.period} ${activePoint.clock}`}
                </text>
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
    </GameSection>
  );
}
