import { Outlet } from "react-router-dom";
import { HomeNav } from "@/components/home/HomeNav";
import { LiveTicker } from "@/components/home/LiveTicker";
import { SiteFooter } from "@/components/SiteFooter";
import { useWnbaScoreboard } from "@/hooks/useWnbaScoreboard";

export function HomeChromeLayout() {
  const { tickerGames, hasNeverLoaded } = useWnbaScoreboard();
  return (
    <div className="flex min-h-screen flex-col bg-black text-white">
      <HomeNav />
      <LiveTicker games={tickerGames} isError={hasNeverLoaded} />
      <main className="flex-1">
        <Outlet />
      </main>
      <SiteFooter />
    </div>
  );
}
