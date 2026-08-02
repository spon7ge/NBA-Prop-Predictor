export type CalloutId = "line" | "odds" | "edge" | "ev";

export const DEMO_PROP = {
  playerName: "LeBron James",
  teamAbbrev: "LAL",
  position: "F",
  matchup: "DEN vs LAL",
  tip: "Tue 7:00pm",
  stat: "Points",
  line: 22.5,
  oddsAmerican: -110,
  model: 24.7,
  ev: 4,
  side: "Over" as const,
  bookLabel: "FanDuel",
} as const;

export const CALLOUTS: Record<CalloutId, { title: string; body: string }> = {
  line: {
    title: "The number to beat",
    body: "22.5 is the points line. Over means you think he’ll score more than 22.5 in the game.",
  },
  odds: {
    title: "What −110 means",
    body: "A minus means this side is the favorite — you risk more than you win. −110 means bet $110 to profit $100.",
  },
  edge: {
    title: "Our model’s guess",
    body: "We project 24.7 points based on the information fed into our model.",
  },
  ev: {
    title: "What EV means",
    body: "Expected value tells you whether a bet is a good deal on average — not whether it'll win this time, but whether making bets like it over and over would leave you ahead.",
  },
};

export function formatEvPercent(ev: number): string {
  if (ev > 0) return `+${ev}%`;
  if (ev < 0) return `−${Math.abs(ev)}%`;
  return "0%";
}
