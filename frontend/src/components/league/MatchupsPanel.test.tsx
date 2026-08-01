import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { MatchupsPanel } from "./MatchupsPanel";
import type { MatchupGame } from "./types";

const live: MatchupGame = {
  id: "live-1",
  espnEventId: "1",
  league: "wnba",
  status: "live",
  statusLabel: "Q4 3:31",
  away: { abbrev: "GS", name: "Golden State Valkyries", score: 77 },
  home: { abbrev: "PHX", name: "Phoenix Mercury", score: 78 },
};

const finalGame: MatchupGame = {
  id: "final-1",
  espnEventId: "2",
  league: "wnba",
  status: "final",
  statusLabel: "Final",
  away: { abbrev: "ATL", name: "Atlanta Dream", score: 82 },
  home: { abbrev: "DAL", name: "Dallas Wings", score: 81 },
};

const defaultNavProps = {
  selectedDate: "2026-08-01",
  todayDate: "2026-08-01",
  onPrevDay: () => {},
  onNextDay: () => {},
  onGoToday: () => {},
};

function renderPanel(games: MatchupGame[], props = {}) {
  return render(
    <MemoryRouter>
      <MatchupsPanel games={games} {...defaultNavProps} {...props} />
    </MemoryRouter>,
  );
}

describe("MatchupsPanel", () => {
  it("splits live and rest and shows count", () => {
    renderPanel([live, finalGame]);
    expect(screen.getByRole("heading", { name: "Matchups" })).toBeInTheDocument();
    expect(
      screen.getByText(
        "2 games · open a card for box score, play-by-play & win probability",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("Live now")).toBeInTheDocument();
    expect(screen.getByText("Rest of the slate")).toBeInTheDocument();
    expect(screen.getByText("Live now").nextElementSibling).toHaveClass(
      "grid",
      "md:grid-cols-2",
    );
    expect(screen.getByText("Rest of the slate").nextElementSibling).toHaveClass(
      "grid",
      "md:grid-cols-2",
    );
  });

  it("enables day navigation and returns to today from center control", async () => {
    const user = userEvent.setup();
    const onPrevDay = vi.fn();
    const onNextDay = vi.fn();
    const onGoToday = vi.fn();
    renderPanel([live], {
      selectedDate: "2026-07-28",
      todayDate: "2026-08-01",
      onPrevDay,
      onNextDay,
      onGoToday,
    });
    expect(screen.getByRole("button", { name: /previous day/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /next day/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Jul 28" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /previous day/i }));
    await user.click(screen.getByRole("button", { name: /next day/i }));
    await user.click(screen.getByRole("button", { name: "Jul 28" }));
    expect(onPrevDay).toHaveBeenCalledOnce();
    expect(onNextDay).toHaveBeenCalledOnce();
    expect(onGoToday).toHaveBeenCalledOnce();
  });

  it("shows Today label on the slate date", () => {
    renderPanel([live], {
      selectedDate: "2026-08-01",
      todayDate: "2026-08-01",
      onPrevDay: () => {},
      onNextDay: () => {},
      onGoToday: () => {},
    });
    expect(screen.getByRole("button", { name: "Today" })).toBeInTheDocument();
  });

  it("shows empty slate copy when no games", () => {
    renderPanel([], {
      selectedDate: "2026-07-28",
      todayDate: "2026-08-01",
      onPrevDay: () => {},
      onNextDay: () => {},
      onGoToday: () => {},
    });
    expect(screen.getByText("No games on this slate")).toBeInTheDocument();
  });

  it("hides Live now when no in-progress games", () => {
    renderPanel([finalGame]);
    expect(screen.queryByText("Live now")).not.toBeInTheDocument();
    expect(screen.getByText("Rest of the slate")).toBeInTheDocument();
  });

  it("announces an error when matchups cannot load", () => {
    renderPanel([], { isError: true });
    expect(screen.getByRole("status")).toHaveTextContent(
      "Unable to load matchups",
    );
  });
});
