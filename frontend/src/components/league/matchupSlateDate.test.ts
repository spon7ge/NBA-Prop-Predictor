import { describe, expect, it } from "vitest";
import {
  formatMatchupNavLabel,
  isValidEtDate,
  parseMatchupDateParam,
  shiftEtDate,
  slateEtDate,
} from "./matchupSlateDate";

describe("matchupSlateDate", () => {
  it("uses ET calendar date and lags before 3:00 AM ET", () => {
    // 2026-08-01 02:30 ET = 06:30 UTC
    expect(slateEtDate(new Date("2026-08-01T06:30:00Z"))).toBe("2026-07-31");
    // 2026-08-01 03:00 ET = 07:00 UTC
    expect(slateEtDate(new Date("2026-08-01T07:00:00Z"))).toBe("2026-08-01");
  });

  it("shifts YYYY-MM-DD by calendar days", () => {
    expect(shiftEtDate("2026-07-31", 1)).toBe("2026-08-01");
    expect(shiftEtDate("2026-08-01", -1)).toBe("2026-07-31");
  });

  it("validates and parses date params", () => {
    expect(isValidEtDate("2026-07-28")).toBe(true);
    expect(isValidEtDate("07-28-2026")).toBe(false);
    expect(parseMatchupDateParam("2026-07-28", "2026-08-01")).toBe("2026-07-28");
    expect(parseMatchupDateParam("nope", "2026-08-01")).toBe("2026-08-01");
    expect(parseMatchupDateParam(null, "2026-08-01")).toBe("2026-08-01");
  });

  it("labels today vs short month day", () => {
    expect(formatMatchupNavLabel("2026-08-01", "2026-08-01")).toBe("Today");
    expect(formatMatchupNavLabel("2026-07-28", "2026-08-01")).toBe("Jul 28");
  });
});
