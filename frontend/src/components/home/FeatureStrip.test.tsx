import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { FeatureStrip } from "./FeatureStrip";

describe("FeatureStrip", () => {
  it("renders clarity features", () => {
    render(<FeatureStrip />);
    expect(
      screen.getByRole("heading", { name: /built to get you ready/i }),
    ).toBeInTheDocument();
    expect(screen.getByText("Follow")).toBeInTheDocument();
    expect(screen.getByText("Understand")).toBeInTheDocument();
    expect(screen.getByText("Decide")).toBeInTheDocument();
  });
});
