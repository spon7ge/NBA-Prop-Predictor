import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { PropExplainerCallouts } from "./PropExplainerCallouts";

describe("PropExplainerCallouts", () => {
  it("shows all callout titles on mobile", () => {
    render(<PropExplainerCallouts selectedSide="over" layout="mobile" />);
    expect(screen.getByText("The number to beat")).toBeInTheDocument();
    expect(screen.getByText("Pick the side")).toBeInTheDocument();
    expect(screen.getByText("Model edge")).toBeInTheDocument();
    expect(screen.getByText("Why EV flipped")).toBeInTheDocument();
  });

  it("marks emphasized callouts for Over via data attribute", () => {
    render(<PropExplainerCallouts selectedSide="over" layout="mobile" />);
    expect(screen.getByTestId("callout-line")).toHaveAttribute(
      "data-emphasized",
      "true",
    );
    expect(screen.getByTestId("callout-side")).toHaveAttribute(
      "data-emphasized",
      "true",
    );
    expect(screen.getByTestId("callout-edge")).toHaveAttribute(
      "data-emphasized",
      "false",
    );
    expect(screen.getByTestId("callout-flip")).toHaveAttribute(
      "data-emphasized",
      "false",
    );
  });

  it("desktop left slot shows line+edge with Under emphasis", () => {
    render(
      <PropExplainerCallouts
        selectedSide="under"
        layout="desktop"
        slot="left"
      />,
    );
    expect(screen.getByTestId("callout-line")).toHaveAttribute(
      "data-emphasized",
      "false",
    );
    expect(screen.getByTestId("callout-edge")).toHaveAttribute(
      "data-emphasized",
      "true",
    );
    expect(screen.queryByTestId("callout-side")).not.toBeInTheDocument();
  });

  it("desktop right slot shows side+flip", () => {
    render(
      <PropExplainerCallouts
        selectedSide="under"
        layout="desktop"
        slot="right"
      />,
    );
    expect(screen.getByTestId("callout-flip")).toHaveAttribute(
      "data-emphasized",
      "true",
    );
    expect(screen.queryByTestId("callout-line")).not.toBeInTheDocument();
  });
});
