import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { BrandHero } from "./BrandHero";

describe("BrandHero", () => {
  beforeEach(() => {
    window.matchMedia = vi.fn().mockImplementation(() => ({
      matches: false,
      media: "",
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders brand copy and CTA to live now", () => {
    render(<BrandHero />);
    expect(
      screen.getByRole("heading", { name: /hoopvista/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/season pass/i)).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: /see what.?s live/i }),
    ).toHaveAttribute("href", "#live-now");
  });
});
