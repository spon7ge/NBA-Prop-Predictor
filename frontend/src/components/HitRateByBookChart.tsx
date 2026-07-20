import type { ApiBookDailyTrend } from "@/types/api";
import type { Book } from "@/types/slate";
import { BOOK_LABELS, BOOKS } from "@/lib/constants";

const BOOK_DISPLAY_TO_SLUG: Record<string, Book> = {
  prizepicks: "prizepicks",
  underdog: "underdog",
  draftkings: "draftkings",
  "draftkings pick6": "draftkings",
  "draftkings pick 6": "draftkings",
  betr: "betr",
  "betr dfs": "betr",
};

const BOOK_STROKE: Record<Book, string> = {
  prizepicks: "#7eb8ff",
  underdog: "#f0c574",
  draftkings: "#6fd48a",
  betr: "#e0a0d0",
};

const FALLBACK_STROKES = ["#a8c4a0", "#c4a882", "#9aabb8", "#c9a0b4"];

interface HitRateByBookChartProps {
  series: ApiBookDailyTrend[];
  /** Short label for the y-metric (props hit rate vs cash rate). */
  metricLabel?: string;
}

function bookSlugFromDisplay(raw: string): Book | null {
  const key = raw.trim().toLowerCase();
  if (key in BOOK_DISPLAY_TO_SLUG) return BOOK_DISPLAY_TO_SLUG[key];
  for (const book of BOOKS) {
    if (key.includes(book)) return book;
  }
  return null;
}

function bookLabel(raw: string): string {
  const slug = bookSlugFromDisplay(raw);
  return slug ? BOOK_LABELS[slug] : raw;
}

function strokeFor(raw: string, index: number): string {
  const slug = bookSlugFromDisplay(raw);
  if (slug) return BOOK_STROKE[slug];
  return FALLBACK_STROKES[index % FALLBACK_STROKES.length]!;
}

function formatShortDate(iso: string): string {
  const parts = iso.split("-");
  if (parts.length < 3) return iso;
  const month = Number(parts[1]);
  const day = Number(parts[2]);
  if (!month || !day) return iso;
  return `${month}/${day}`;
}

function pctLabel(rate: number | null): string {
  if (rate == null || Number.isNaN(rate)) return "—";
  return `${Math.round(rate * 100)}%`;
}

export function HitRateByBookChart({
  series,
  metricLabel = "Hit rate",
}: HitRateByBookChartProps) {
  const active = series.filter((s) => s.points.some((p) => p.n > 0));
  if (active.length === 0) {
    return (
      <p className="results-viz-empty">No daily hit-rate data for this window.</p>
    );
  }

  const dates = Array.from(
    new Set(active.flatMap((s) => s.points.filter((p) => p.n > 0).map((p) => p.game_date))),
  ).sort();

  const w = 360;
  const h = 120;
  const padL = 28;
  const padR = 10;
  const padT = 12;
  const padB = 22;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  function xAt(date: string): number {
    if (dates.length <= 1) return padL + plotW / 2;
    const i = dates.indexOf(date);
    return padL + (i / (dates.length - 1)) * plotW;
  }

  function yAt(rate: number): number {
    return padT + (1 - rate) * plotH;
  }

  const yTicks = [0, 0.5, 1];

  return (
    <div className="results-hitrate-chart">
      <div className="results-hitrate-chart-head">
        <span className="results-roi-stat-label">{metricLabel} by book</span>
        <span className="results-hitrate-chart-hint">Daily</span>
      </div>
      <svg
        className="results-hitrate-svg"
        viewBox={`0 0 ${w} ${h}`}
        role="img"
        aria-label={`${metricLabel} by bookmaker over ${dates.length} days`}
      >
        {yTicks.map((t) => {
          const y = yAt(t);
          return (
            <g key={t}>
              <line
                className="results-hitrate-grid"
                x1={padL}
                y1={y}
                x2={w - padR}
                y2={y}
              />
              <text
                className="results-hitrate-axis"
                x={padL - 4}
                y={y + 3}
                textAnchor="end"
              >
                {Math.round(t * 100)}%
              </text>
            </g>
          );
        })}
        {dates.map((d) => (
          <text
            key={d}
            className="results-hitrate-axis"
            x={xAt(d)}
            y={h - 4}
            textAnchor="middle"
          >
            {formatShortDate(d)}
          </text>
        ))}
        {active.map((s, si) => {
          const color = strokeFor(s.bookmaker, si);
          const pts = s.points
            .filter((p) => p.n > 0 && p.hit_rate != null)
            .sort((a, b) => a.game_date.localeCompare(b.game_date));
          if (pts.length === 0) return null;
          const line = pts
            .map((p) => `${xAt(p.game_date)},${yAt(p.hit_rate!)}`)
            .join(" ");
          return (
            <g key={s.bookmaker}>
              <polyline
                className="results-hitrate-line"
                points={line}
                fill="none"
                stroke={color}
                strokeWidth="2.25"
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{ animationDelay: `${si * 0.08}s` }}
              />
              {pts.map((p) => (
                <circle
                  key={`${s.bookmaker}-${p.game_date}`}
                  className="results-hitrate-dot"
                  cx={xAt(p.game_date)}
                  cy={yAt(p.hit_rate!)}
                  r="3.2"
                  fill={color}
                  style={{ animationDelay: `${0.35 + si * 0.08}s` }}
                >
                  <title>
                    {bookLabel(s.bookmaker)} · {formatShortDate(p.game_date)} ·{" "}
                    {pctLabel(p.hit_rate)} ({p.hits}/{p.n})
                  </title>
                </circle>
              ))}
            </g>
          );
        })}
      </svg>
      <ul className="results-hitrate-legend" aria-label="Bookmakers">
        {active.map((s, si) => (
          <li key={s.bookmaker} className="results-hitrate-legend-item">
            <span
              className="results-hitrate-legend-swatch"
              style={{ background: strokeFor(s.bookmaker, si) }}
              aria-hidden="true"
            />
            <span className="results-hitrate-legend-label">
              {bookLabel(s.bookmaker)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
