import { describe, expect, it } from "vitest";
import {
  CALLOUTS,
  DEMO_PROP,
  formatEvPercent,
} from "./propExplainerDemo";

describe("propExplainerDemo", () => {
  it("exposes the LeBron demo numbers", () => {
    expect(DEMO_PROP.playerName).toBe("LeBron James");
    expect(DEMO_PROP.teamAbbrev).toBe("LAL");
    expect(DEMO_PROP.position).toBe("F");
    expect(DEMO_PROP.stat).toBe("Points");
    expect(DEMO_PROP.line).toBe(22.5);
    expect(DEMO_PROP.oddsAmerican).toBe(-110);
    expect(DEMO_PROP.model).toBe(24.7);
    expect(DEMO_PROP.ev).toBe(4);
    expect(DEMO_PROP.side).toBe("Over");
    expect(DEMO_PROP.matchup).toBe("DEN vs LAL");
    expect(DEMO_PROP.tip).toBe("Tue 7:00pm");
  });

  it("formats EV with sign and Unicode minus for negatives", () => {
    expect(formatEvPercent(4)).toBe("+4%");
    expect(formatEvPercent(-4)).toBe("−4%");
  });

  it("teaches line, odds, model, and EV in plain language", () => {
    expect(CALLOUTS.line.body).toMatch(/more than 22\.5/i);
    expect(CALLOUTS.odds.body).toMatch(/bet \$110 to profit \$100/i);
    expect(CALLOUTS.odds.body).toMatch(/favorite/i);
    expect(CALLOUTS.edge.body).toMatch(/fed into our model/i);
    expect(CALLOUTS.ev.body).toMatch(/expected value/i);
  });
});
