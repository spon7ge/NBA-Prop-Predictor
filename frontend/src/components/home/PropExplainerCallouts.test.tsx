import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { PropExplainerCallouts } from "./PropExplainerCallouts";

describe("PropExplainerCallouts", () => {
  it("shows all teaching callout titles on mobile", () => {
    render(<PropExplainerCallouts layout="mobile" />);
    expect(screen.getByText("The number to beat")).toBeInTheDocument();
    expect(screen.getByText("What −110 means")).toBeInTheDocument();
    expect(screen.getByText("Our model’s guess")).toBeInTheDocument();
    expect(screen.getByText("What EV means")).toBeInTheDocument();
  });

  it("desktop left slot shows line and model", () => {
    render(<PropExplainerCallouts layout="desktop" slot="left" />);
    expect(screen.getByTestId("callout-line")).toBeInTheDocument();
    expect(screen.getByTestId("callout-edge")).toBeInTheDocument();
    expect(screen.queryByTestId("callout-odds")).not.toBeInTheDocument();
  });

  it("desktop right slot shows odds and EV", () => {
    render(<PropExplainerCallouts layout="desktop" slot="right" />);
    expect(screen.getByTestId("callout-odds")).toBeInTheDocument();
    expect(screen.getByTestId("callout-ev")).toBeInTheDocument();
    expect(screen.queryByTestId("callout-line")).not.toBeInTheDocument();
  });
});
