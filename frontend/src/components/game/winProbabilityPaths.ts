const CHART_WIDTH = 640;
const CHART_HEIGHT = 112;
const CHART_PAD_LEFT = 20;
const CHART_PAD_RIGHT = 72;
const CHART_PAD_TOP = 8;
const CHART_PAD_BOTTOM = 4;
export const PLOT_WIDTH = CHART_WIDTH - CHART_PAD_LEFT - CHART_PAD_RIGHT;
export const PLOT_HEIGHT = CHART_HEIGHT - CHART_PAD_TOP - CHART_PAD_BOTTOM;

export const CHART_GEOMETRY = {
  width: CHART_WIDTH,
  height: CHART_HEIGHT,
  padLeft: CHART_PAD_LEFT,
  padRight: CHART_PAD_RIGHT,
  padTop: CHART_PAD_TOP,
  padBottom: CHART_PAD_BOTTOM,
  plotWidth: PLOT_WIDTH,
  plotHeight: PLOT_HEIGHT,
  yLabelX: CHART_PAD_LEFT - 8,
} as const;

export type SeriesKey = "away" | "home";

type PctPoint = { awayWinPct: number; homeWinPct: number };

export function xForIndex(index: number, count: number): number {
  if (count <= 1) return CHART_PAD_LEFT + PLOT_WIDTH / 2;
  return CHART_PAD_LEFT + (index / (count - 1)) * PLOT_WIDTH;
}

export function yForPct(pct: number): number {
  return CHART_PAD_TOP + PLOT_HEIGHT - (pct / 100) * PLOT_HEIGHT;
}

export function nearestIndexForClientX(
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

export function buildSeriesPathD(
  points: PctPoint[],
  series: SeriesKey,
  fromIndex: number,
  toIndex: number,
): string {
  if (points.length === 0) return "";
  if (fromIndex > toIndex) return "";
  if (fromIndex < 0 || toIndex >= points.length) return "";

  const coords = [];
  for (let i = fromIndex; i <= toIndex; i++) {
    const pct =
      series === "home" ? points[i].homeWinPct : points[i].awayWinPct;
    coords.push({ x: xForIndex(i, points.length), y: yForPct(pct) });
  }

  return coords
    .map((c, i) => `${i === 0 ? "M" : "L"}${c.x} ${c.y}`)
    .join(" ");
}

export function buildSplitSeriesPaths(
  points: PctPoint[],
  activeIndex: number,
): {
  awayVivid: string;
  awayMuted: string;
  homeVivid: string;
  homeMuted: string;
} {
  if (points.length === 0) {
    return { awayVivid: "", awayMuted: "", homeVivid: "", homeMuted: "" };
  }
  const scrub = Math.min(Math.max(activeIndex, 0), points.length - 1);
  const last = points.length - 1;
  return {
    awayVivid: buildSeriesPathD(points, "away", 0, scrub),
    homeVivid: buildSeriesPathD(points, "home", 0, scrub),
    awayMuted:
      scrub < last ? buildSeriesPathD(points, "away", scrub, last) : "",
    homeMuted:
      scrub < last ? buildSeriesPathD(points, "home", scrub, last) : "",
  };
}
