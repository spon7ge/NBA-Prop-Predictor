export type PropSide = "over" | "under";
export type CalloutId = "line" | "side" | "edge" | "flip";

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
  evOver: 4,
  evUnder: -4,
  bookLabel: "FanDuel",
} as const;

export const CALLOUTS: Record<CalloutId, { title: string; body: string }> = {
  line: {
    title: "The number to beat",
    body: "FanDuel lists 22.5 points at −110. That’s the line you’re betting against.",
  },
  side: {
    title: "Pick the side",
    body: "Over — he clears 22.5. Under — he stays under. Active side drives the EV shown.",
  },
  edge: {
    title: "Model edge",
    body: "Projection 24.7 vs line 22.5 → about +4% EV on Over.",
  },
  flip: {
    title: "Why EV flipped",
    body: "Same line and model — you just chose the side the projection doesn’t favor.",
  },
};

export function evForSide(side: PropSide): number {
  return side === "over" ? DEMO_PROP.evOver : DEMO_PROP.evUnder;
}

export function formatEvPercent(ev: number): string {
  if (ev > 0) return `+${ev}%`;
  if (ev < 0) return `−${Math.abs(ev)}%`;
  return "0%";
}

export function isCalloutEmphasized(id: CalloutId, side: PropSide): boolean {
  if (side === "over") return id === "line" || id === "side";
  return id === "edge" || id === "flip";
}
