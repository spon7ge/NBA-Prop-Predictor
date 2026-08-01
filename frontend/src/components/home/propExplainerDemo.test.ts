import { describe, expect, it } from "vitest";
import {
  DEMO_PROP,
  evForSide,
  formatEvPercent,
  isCalloutEmphasized,
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
    expect(DEMO_PROP.evOver).toBe(4);
    expect(DEMO_PROP.evUnder).toBe(-4);
    expect(DEMO_PROP.matchup).toBe("DEN vs LAL");
    expect(DEMO_PROP.tip).toBe("Tue 7:00pm");
  });

  it("returns EV for the selected side", () => {
    expect(evForSide("over")).toBe(4);
    expect(evForSide("under")).toBe(-4);
  });

  it("formats EV with sign and Unicode minus for negatives", () => {
    expect(formatEvPercent(4)).toBe("+4%");
    expect(formatEvPercent(-4)).toBe("−4%");
  });

  it("emphasizes callouts per side rules", () => {
    expect(isCalloutEmphasized("line", "over")).toBe(true);
    expect(isCalloutEmphasized("side", "over")).toBe(true);
    expect(isCalloutEmphasized("edge", "over")).toBe(false);
    expect(isCalloutEmphasized("flip", "over")).toBe(false);

    expect(isCalloutEmphasized("edge", "under")).toBe(true);
    expect(isCalloutEmphasized("flip", "under")).toBe(true);
    expect(isCalloutEmphasized("line", "under")).toBe(false);
    expect(isCalloutEmphasized("side", "under")).toBe(false);
  });
});
