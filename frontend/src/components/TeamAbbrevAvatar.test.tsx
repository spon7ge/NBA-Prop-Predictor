import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { TeamAbbrevAvatar } from "./TeamAbbrevAvatar";

describe("TeamAbbrevAvatar", () => {
  it("renders a logo image when logoUrl is set", () => {
    const { container } = render(
      <TeamAbbrevAvatar
        abbrev="ATL"
        logoUrl="https://a.espncdn.com/i/teamlogos/wnba/500/atl.png"
      />,
    );
    const img = container.querySelector("img");
    expect(img).toHaveAttribute(
      "src",
      "https://a.espncdn.com/i/teamlogos/wnba/500/atl.png",
    );
    expect(img).toHaveAttribute("alt", "");
  });

  it("renders the abbrev letter when logoUrl is null", () => {
    render(<TeamAbbrevAvatar abbrev="ATL" logoUrl={null} />);
    expect(screen.getByText("A")).toBeInTheDocument();
    expect(screen.queryByRole("presentation")).not.toBeInTheDocument();
  });

  it("falls back to the letter after image error", () => {
    const { container } = render(
      <TeamAbbrevAvatar
        abbrev="ATL"
        logoUrl="https://a.espncdn.com/i/teamlogos/wnba/500/atl.png"
      />,
    );
    fireEvent.error(container.querySelector("img")!);
    expect(screen.getByText("A")).toBeInTheDocument();
  });
});
