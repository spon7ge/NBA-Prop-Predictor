import { Outlet } from "react-router-dom";
import { HomeNav } from "@/components/home/HomeNav";
import { LiveTicker } from "@/components/home/LiveTicker";
import { useWnbaScoreboard } from "@/hooks/useWnbaScoreboard";

export function HomeChromeLayout() {
  const { tickerGames } = useWnbaScoreboard();
  return (
    <div className="min-h-screen bg-black text-white">
      <HomeNav />
      <LiveTicker games={tickerGames} />
      <main>
        <Outlet />
      </main>
    </div>
  );
}
