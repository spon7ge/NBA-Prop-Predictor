import { useEffect, useState } from "react";
import type { View } from "@/types/slate";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import { TopLegsView } from "@/components/TopLegsView";
import { AllPlayersView } from "@/components/AllPlayersView";

function getInitialView(): View {
  try {
    const q = new URLSearchParams(window.location.search);
    const viewParam = q.get("view");
    if (viewParam === "players" || viewParam === "pairs") return viewParam;
  } catch {
    /* ignore */
  }
  return "pairs";
}

export default function App() {
  const [activeView, setActiveView] = useState<View>(getInitialView);

  useEffect(() => {
    const url = new URL(window.location.href);
    url.searchParams.set("view", activeView);
    window.history.replaceState(null, "", url);
  }, [activeView]);

  return (
    <div className="page">
      <Header activeView={activeView} onViewChange={setActiveView} />
      {activeView === "pairs" ? <TopLegsView /> : <AllPlayersView />}
      <Footer />
    </div>
  );
}
