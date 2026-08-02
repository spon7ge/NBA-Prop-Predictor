import { Outlet } from "react-router-dom";
import { HomeNav } from "@/components/home/HomeNav";
import { LiveTicker } from "@/components/home/LiveTicker";
import { mergeLeagueScoreboards } from "@/components/home/mergeLeagueScoreboards";
import { SiteFooter } from "@/components/SiteFooter";
import { useMlbScoreboard } from "@/hooks/useMlbScoreboard";
import { useWnbaScoreboard } from "@/hooks/useWnbaScoreboard";

export function HomeChromeLayout() {
  const wnba = useWnbaScoreboard();
  const mlb = useMlbScoreboard();
  const { tickerGames, hasNeverLoaded } = mergeLeagueScoreboards([wnba, mlb]);
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
