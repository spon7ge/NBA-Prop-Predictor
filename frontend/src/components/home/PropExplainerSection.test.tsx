import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { PropExplainerSection } from "./PropExplainerSection";

function renderSection() {
  return render(
    <MemoryRouter>
      <PropExplainerSection />
    </MemoryRouter>,
  );
}

describe("PropExplainerSection", () => {
  it("renders heading and CTA to live props", () => {
    renderSection();
    expect(
      screen.getByRole("heading", { name: /read the line\. see the edge/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: /see live props/i }),
    ).toHaveAttribute("href", "/wnba/prop_picks");
  });

  it("shows Over example with positive EV and odds teaching", () => {
    renderSection();
    expect(screen.getAllByText("+4%").length).toBeGreaterThan(0);
    expect(screen.getAllByText("What −110 means").length).toBeGreaterThan(0);
    expect(screen.queryByRole("button", { name: /under/i })).not.toBeInTheDocument();
  });
});
