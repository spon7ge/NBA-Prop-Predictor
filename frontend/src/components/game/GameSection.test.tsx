import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { GAME_SECTION_SURFACE, GameSection } from "./GameSection";

describe("GameSection", () => {
  it("applies the quiet Live-now surface classes", () => {
    render(
      <GameSection>
        <h2>Shot chart</h2>
      </GameSection>,
    );
    const section = screen.getByRole("heading", { name: "Shot chart" }).closest(
      "section",
    );
    expect(section).toHaveClass(
      "rounded-xl",
      "border",
      "border-white/10",
      "bg-white/[0.03]",
      "p-4",
    );
  });

  it("exports GAME_SECTION_SURFACE matching the Live-now card treatment", () => {
    expect(GAME_SECTION_SURFACE).toBe(
      "rounded-xl border border-white/10 bg-white/[0.03] p-4",
    );
  });

  it("merges an optional className", () => {
    render(
      <GameSection className="space-y-3">
        <span>inner</span>
      </GameSection>,
    );
    expect(screen.getByText("inner").closest("section")).toHaveClass(
      "space-y-3",
      "rounded-xl",
    );
  });
});
