import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { HomePage } from "./HomePage";

function renderHome() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <HomePage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("HomePage", () => {
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
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ date: "2026-08-01", fetched_at: "", games: [] }),
      }),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("renders the Apple-style section stack", () => {
    renderHome();
    expect(
      screen.getByRole("heading", { name: /hoopvista/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: /see what.?s live/i }),
    ).toHaveAttribute("href", "#live-now");
    expect(
      screen.getByRole("heading", { name: /live now/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /stories/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", {
        name: /the stats site\. the betting edge\. together/i,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /enter a league/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: /learn the game/i }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: /^explore$/i }),
    ).not.toBeInTheDocument();
  });
});
