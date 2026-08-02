import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { LeagueFuturesPage } from "./LeagueFuturesPage";

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={["/wnba/futures"]}>
        <LeagueFuturesPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("LeagueFuturesPage", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders futures from API with active Futures subnav", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        season: 2026,
        as_of: "2026-08-01T00:00:00Z",
        markets: [
          {
            id: "8146",
            name: "WNBA - Winner",
            display_name: "Finals Winner",
            provider: "DraftKings",
            entries: [
              {
                team_id: "8",
                abbrev: "NYL",
                name: "New York Liberty",
                logo_url: null,
                odds_american: "+250",
              },
            ],
          },
        ],
        error: null,
      }),
    });

    renderPage();
    expect(await screen.findByText("Finals Winner")).toBeInTheDocument();
    expect(screen.getByText("New York Liberty")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Futures" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });
});
