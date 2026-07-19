import { useEffect, useState } from "react";
import type { ApiLeagueFilter } from "@/types/api";
import type { View } from "@/types/slate";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import { TopLegsView } from "@/components/TopLegsView";
import { AllPlayersView } from "@/components/AllPlayersView";
import { ResultsView } from "@/components/ResultsView";

function getInitialView(): View {
  try {
    const q = new URLSearchParams(window.location.search);
    const viewParam = q.get("view");
    if (viewParam === "players" || viewParam === "pairs" || viewParam === "results") {
      return viewParam;
    }
  } catch {
    /* ignore */
  }
  return "pairs";
}

function getInitialLeague(): ApiLeagueFilter {
  try {
    const q = new URLSearchParams(window.location.search);
    const league = q.get("league");
    if (league === "nba" || league === "wnba") return league;
  } catch {
    /* ignore */
  }
  return "wnba";
}

export default function App() {
  const [activeView, setActiveView] = useState<View>(getInitialView);
  const [league, setLeague] = useState<ApiLeagueFilter>(getInitialLeague);

  useEffect(() => {
    const url = new URL(window.location.href);
    url.searchParams.set("view", activeView);
    url.searchParams.set("league", league);
    window.history.replaceState(null, "", url);
  }, [activeView, league]);

  return (
    <div className="page">
      <Header activeView={activeView} onViewChange={setActiveView} />
      {activeView === "pairs" ? (
        <TopLegsView league={league} onLeagueChange={setLeague} />
      ) : activeView === "players" ? (
        <AllPlayersView league={league} onLeagueChange={setLeague} />
      ) : (
        <ResultsView league={league} onLeagueChange={setLeague} />
      )}
      <Footer />
    </div>
  );
}
