import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { PropExplainerCard } from "./PropExplainerCard";

describe("PropExplainerCard", () => {
  it("renders player, line, model, FanDuel, EV, and Over example", () => {
    render(<PropExplainerCard />);
    expect(screen.getByText("LeBron James")).toBeInTheDocument();
    expect(screen.getByText("LAL · F")).toBeInTheDocument();
    expect(screen.getByText("DEN vs LAL")).toBeInTheDocument();
    expect(screen.getByText("Tue 7:00pm")).toBeInTheDocument();
    expect(screen.getByText("22.5")).toBeInTheDocument();
    expect(screen.getByText("Points")).toBeInTheDocument();
    expect(screen.getByText("24.7")).toBeInTheDocument();
    expect(screen.getByText("+4%")).toBeInTheDocument();
    expect(screen.getByText(/−110/)).toBeInTheDocument();
    expect(screen.getByLabelText(/example side over/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /under/i })).not.toBeInTheDocument();
  });
});
