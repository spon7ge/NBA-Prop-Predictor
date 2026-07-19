import { formatUsd, type BankrollPoint } from "@/lib/legsRoi";

interface BankrollCurveProps {
  points: BankrollPoint[];
  startBankroll: number;
}

export function BankrollCurve({ points, startBankroll }: BankrollCurveProps) {
  if (points.length === 0) {
    return (
      <p className="results-viz-empty">No decided tickets yet for a bankroll path.</p>
    );
  }

  const w = 320;
  const h = 88;
  const padX = 8;
  const padY = 10;
  const values = [startBankroll, ...points.map((p) => p.bankroll)];
  const minV = Math.min(...values);
  const maxV = Math.max(...values);
  const span = Math.max(maxV - minV, 1);
  const end = points[points.length - 1]!;
  const up = end.bankroll >= startBankroll;

  function xAt(i: number, n: number): number {
    if (n <= 1) return w / 2;
    return padX + (i / (n - 1)) * (w - padX * 2);
  }

  function yAt(v: number): number {
    return padY + (1 - (v - minV) / span) * (h - padY * 2);
  }

  const coords = points.map((p, i) => ({
    x: xAt(i, points.length),
    y: yAt(p.bankroll),
    p,
  }));

  const line = coords.map((c) => `${c.x},${c.y}`).join(" ");
  const area = [
    `${coords[0]!.x},${h - padY}`,
    ...coords.map((c) => `${c.x},${c.y}`),
    `${coords[coords.length - 1]!.x},${h - padY}`,
  ].join(" ");

  return (
    <div className="results-bankroll-curve">
      <div className="results-bankroll-curve-head">
        <span className="results-roi-stat-label">Bankroll path</span>
        <span
          className={`results-bankroll-curve-end${
            up ? " results-roi-stat-value--pos" : " results-roi-stat-value--neg"
          }`}
        >
          {formatUsd(startBankroll)} → {formatUsd(end.bankroll)}
        </span>
      </div>
      <svg
        className="results-bankroll-svg"
        viewBox={`0 0 ${w} ${h}`}
        role="img"
        aria-label={`Bankroll from ${formatUsd(startBankroll)} to ${formatUsd(end.bankroll)} over ${points.length} tickets`}
      >
        <defs>
          <linearGradient id="bankFill" x1="0" y1="0" x2="0" y2="1">
            <stop
              offset="0%"
              stopColor={up ? "rgba(111,212,138,0.35)" : "rgba(224,128,128,0.35)"}
            />
            <stop offset="100%" stopColor="rgba(111,212,138,0)" />
          </linearGradient>
        </defs>
        <polygon className="results-bankroll-area" points={area} fill="url(#bankFill)" />
        <polyline
          className={`results-bankroll-line${up ? " results-bankroll-line--up" : " results-bankroll-line--down"}`}
          points={line}
          fill="none"
          strokeWidth="2.25"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {coords.map((c) => (
          <circle
            key={c.p.index}
            className={
              c.p.cashed
                ? "results-bankroll-dot results-bankroll-dot--hit"
                : "results-bankroll-dot results-bankroll-dot--miss"
            }
            cx={c.x}
            cy={c.y}
            r="3.2"
          >
            <title>
              {c.p.gameDate} · {c.p.label} · {c.p.cashed ? "CASH" : "MISS"} ·{" "}
              {formatUsd(c.p.delta)} → {formatUsd(c.p.bankroll)}
            </title>
          </circle>
        ))}
      </svg>
    </div>
  );
}
