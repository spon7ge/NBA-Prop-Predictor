import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { FeatureStrip } from "./FeatureStrip";

describe("FeatureStrip", () => {
  it("renders clarity features", () => {
    render(<FeatureStrip />);
    expect(
      screen.getByRole("heading", {
        name: /the stats site\. the betting edge\. together/i,
      }),
    ).toBeInTheDocument();
    expect(screen.getByText("Reference")).toBeInTheDocument();
    expect(screen.getByText("Lines")).toBeInTheDocument();
    expect(screen.getByText("Projections")).toBeInTheDocument();
  });
});
