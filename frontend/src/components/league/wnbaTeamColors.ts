/** Brand colors for stats.wnba.com tricodes on dark UI. */
const WNBA_TEAM_COLORS: Record<string, string> = {
  ATL: "#C8102E",
  CHI: "#4E8FD0",
  CON: "#FC4C02",
  DAL: "#C4D600",
  GSV: "#FFC72C",
  IND: "#FFCD00",
  LAS: "#552583",
  LVA: "#C8102E",
  MIN: "#236192",
  NYL: "#6ECEB2",
  PHO: "#E56020",
  SEA: "#2C5234",
  TOR: "#B4975A",
  WAS: "#E31837",
};

const FALLBACK = "rgba(255,255,255,0.5)";

export function teamColor(abbrev: string): string {
  const key = abbrev.trim().toUpperCase();
  return WNBA_TEAM_COLORS[key] ?? FALLBACK;
}
