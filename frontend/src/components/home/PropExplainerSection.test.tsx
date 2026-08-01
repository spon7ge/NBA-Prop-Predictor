import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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

  it("flips EV when Under is selected", async () => {
    const user = userEvent.setup();
    renderSection();
    expect(screen.getAllByText("+4%").length).toBeGreaterThan(0);
    const unders = screen.getAllByRole("button", { name: /under/i });
    await user.click(unders[0]);
    expect(screen.getAllByText("−4%").length).toBeGreaterThan(0);
  });
});
