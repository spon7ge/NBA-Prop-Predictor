import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SiteFooter } from "./SiteFooter";

describe("SiteFooter", () => {
  it("renders disclaimer and copyright", () => {
    render(<SiteFooter />);
    expect(
      screen.getByText(/informational and entertainment purposes only/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText("© Copyright 2026 HoopVista. All Rights Reserved."),
    ).toBeInTheDocument();
  });
});
