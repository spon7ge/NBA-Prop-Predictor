import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { FeatureStrip } from "./FeatureStrip";

describe("FeatureStrip", () => {
  it("renders clarity features", () => {
    render(<FeatureStrip />);
    expect(
      screen.getByRole("heading", { name: /built for clarity/i }),
    ).toBeInTheDocument();
    expect(screen.getByText("Props")).toBeInTheDocument();
    expect(screen.getByText("Edges")).toBeInTheDocument();
    expect(screen.getByText("Explain")).toBeInTheDocument();
  });
});
