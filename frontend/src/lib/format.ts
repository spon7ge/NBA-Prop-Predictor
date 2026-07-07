export function fmt1(x: number | string | null | undefined): string {
  return Number(x).toFixed(1);
}

export function fmtNumOrDash(x: number | string | null | undefined): string {
  if (x == null || x === "") return "—";
  const n = Number(x);
  if (Number.isNaN(n)) return "—";
  return fmt1(n);
}

export function fmtEv(x: number): string {
  return Number(x).toFixed(2);
}

export function fmtSignedEv(x: number): string {
  const n = Number(x);
  const sign = n >= 0 ? "+" : "-";
  return sign + fmtEv(Math.abs(n));
}

export function fmtOverRatePct(overRate: number | string | null | undefined): string {
  if (overRate == null || overRate === "") return "—";
  const n = Number(overRate);
  if (Number.isNaN(n)) return "—";
  return `${Math.round(n * 100)}%`;
}

export function pct0(x: number): string {
  return String(Math.round(Number(x) * 100));
}

export function ordSuffix(n: number | string | null | undefined): string {
  if (n == null || n === "" || Number.isNaN(Number(n))) return "—";
  const i = Math.floor(Math.abs(Number(n)));
  const j = i % 100;
  if (j >= 11 && j <= 13) return `${i}th`;
  switch (i % 10) {
    case 1:
      return `${i}st`;
    case 2:
      return `${i}nd`;
    case 3:
      return `${i}rd`;
    default:
      return `${i}th`;
  }
}

export function fmtOrdinalRank(n: number | string | null | undefined): string {
  if (n == null || n === "") return "—";
  const v = Number(n);
  if (Number.isNaN(v)) return "—";
  return ordSuffix(v);
}

export function spreadFmt(n: number | string | null | undefined): string {
  if (n == null || n === "" || Number.isNaN(Number(n))) return "—";
  const v = Number(n);
  const sign = v > 0 ? "+" : "";
  return sign + v.toFixed(1);
}

export function hitRate(overRate: number, side: string): number {
  const s = side.toLowerCase();
  return s === "over" ? Number(overRate) : 1 - Number(overRate);
}

export function modelDiffGood(side: string, prediction: number, line: number): boolean {
  const d = Number(prediction) - Number(line);
  const s = side.toLowerCase();
  if (s === "over") return d > 0;
  return d < 0;
}

export function diffDisplay(_side: string, prediction: number, line: number): string {
  const d = Number(prediction) - Number(line);
  const sign = d > 0 ? "+" : "";
  return sign + fmt1(d);
}

export function l10StatAvgLabel(market: string): string {
  const m = String(market || "")
    .trim()
    .toUpperCase();
  if (m === "PTS") return "L10 PTS Avg.";
  if (m === "AST") return "L10 AST Avg.";
  if (m === "REB" || m === "REBS") return "L10 REB Avg.";
  if (!m) return "L10 Stat Avg.";
  return `L10 ${String(market).trim()} Avg.`;
}

export function parseSortNumber(val: unknown): number | null {
  if (val == null || val === "") return null;
  const n = Number(val);
  if (Number.isNaN(n)) return null;
  return n;
}
