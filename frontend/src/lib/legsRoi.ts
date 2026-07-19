import type { ApiGradedParlay } from "@/types/api";
import { dfsNetPayoutMult } from "@/lib/constants";

export interface LegsRoiResult {
  decided: number;
  cashed: number;
  missed: number;
  stakePerTicket: number;
  totalStaked: number;
  /** Net profit in dollars (cashed: +stake×net_mult; miss: −stake). */
  profit: number;
  /** profit / totalStaked */
  roi: number | null;
  endingBankroll: number;
}

export interface BankrollPoint {
  index: number;
  gameDate: string;
  label: string;
  cashed: boolean;
  delta: number;
  bankroll: number;
  bookmaker: string;
  nLegs: number;
}

/**
 * Flat-unit ROI on decided Top Legs using platform net-profit multipliers
 * (same table as ``src.utils.slates``).
 */
export function computeLegsRoi(
  parlays: ApiGradedParlay[],
  bankroll: number,
  stakePerTicket: number,
): LegsRoiResult {
  const stake = Math.max(0, stakePerTicket);
  const start = Math.max(0, bankroll);
  let cashed = 0;
  let missed = 0;
  let profit = 0;

  for (const p of parlays) {
    if (p.cashed == null) continue;
    if (p.cashed) {
      cashed += 1;
      profit += stake * dfsNetPayoutMult(p.bookmaker, p.n_legs);
    } else {
      missed += 1;
      profit -= stake;
    }
  }

  const decided = cashed + missed;
  const totalStaked = decided * stake;
  return {
    decided,
    cashed,
    missed,
    stakePerTicket: stake,
    totalStaked,
    profit,
    roi: totalStaked > 0 ? profit / totalStaked : null,
    endingBankroll: start + profit,
  };
}

/** Chronological bankroll path after each decided ticket. */
export function buildBankrollSeries(
  parlays: ApiGradedParlay[],
  bankroll: number,
  stakePerTicket: number,
): BankrollPoint[] {
  const stake = Math.max(0, stakePerTicket);
  let bal = Math.max(0, bankroll);
  const decided = parlays
    .filter((p) => p.cashed != null)
    .slice()
    .sort((a, b) => {
      const d = String(a.game_date).localeCompare(String(b.game_date));
      if (d !== 0) return d;
      if (a.n_legs !== b.n_legs) return a.n_legs - b.n_legs;
      return a.bookmaker.localeCompare(b.bookmaker);
    });

  return decided.map((p, index) => {
    const delta = p.cashed
      ? stake * dfsNetPayoutMult(p.bookmaker, p.n_legs)
      : -stake;
    bal += delta;
    return {
      index,
      gameDate: String(p.game_date),
      label: `${p.n_legs}-leg ${p.bookmaker}`,
      cashed: Boolean(p.cashed),
      delta,
      bankroll: bal,
      bookmaker: p.bookmaker,
      nLegs: p.n_legs,
    };
  });
}

export function formatUsd(n: number): string {
  const sign = n < 0 ? "-" : "";
  return `${sign}$${Math.abs(n).toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: 2,
  })}`;
}

export function formatRoiPct(roi: number | null): string {
  if (roi == null || Number.isNaN(roi)) return "—";
  const pct = roi * 100;
  const sign = pct > 0 ? "+" : "";
  return `${sign}${pct.toFixed(1)}%`;
}
