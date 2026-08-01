import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { LeagueMatchupsPage } from "./LeagueMatchupsPage";

vi.mock("@/hooks/useWnbaScoreboard", () => ({
  useWnbaScoreboard: (dateEt?: string) => ({
    games: [],
    isLoading: false,
    hasNeverLoaded: false,
    data: { date: dateEt ?? "2026-08-01", games: [], fetched_at: "" },
  }),
}));

vi.mock("@/hooks/useWnbaOdds", () => ({
  useWnbaOdds: () => ({ data: undefined }),
}));

vi.mock("@/components/league/matchupSlateDate", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/components/league/matchupSlateDate")>();
  return { ...actual, slateEtDate: () => "2026-08-01" };
});

function renderAt(path: string) {
  const client = new QueryClient();
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route
            path="/wnba/matchups"
            element={<LeagueMatchupsPage league="wnba" />}
          />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("LeagueMatchupsPage date nav", () => {
  it("writes ?date when moving off today and clears it returning", async () => {
    const user = userEvent.setup();
    renderAt("/wnba/matchups");
    expect(screen.getByRole("button", { name: "Today" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /previous day/i }));
    expect(screen.getByRole("button", { name: "Jul 31" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Jul 31" }));
    expect(screen.getByRole("button", { name: "Today" })).toBeInTheDocument();
  });
});
