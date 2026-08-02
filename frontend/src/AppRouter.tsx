import { Routes, Route } from "react-router-dom";
import { HomeChromeLayout } from "@/layouts/HomeChromeLayout";
import { HomePage } from "@/pages/HomePage";
import { AboutPage } from "@/pages/AboutPage";
import { GameDetailPage } from "@/pages/GameDetailPage";
import { LeagueMatchupsPage } from "@/pages/LeagueMatchupsPage";
import { LeagueLeadersPage } from "@/pages/LeagueLeadersPage";
import { LeagueStandingsPage } from "@/pages/LeagueStandingsPage";
import { LeagueFuturesPage } from "@/pages/LeagueFuturesPage";
import { LeaguePlayerPage } from "@/pages/LeaguePlayerPage";
import { LeaguePropPicksPage } from "@/pages/LeaguePropPicksPage";
import { MlbGameStubPage } from "@/pages/MlbGameStubPage";
import { NotFoundPage } from "@/pages/NotFoundPage";

export function AppRouter() {
  return (
    <Routes>
      <Route element={<HomeChromeLayout />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="/games/:espnEventId" element={<GameDetailPage />} />
        <Route
          path="/wnba/matchups"
          element={<LeagueMatchupsPage league="wnba" />}
        />
        <Route path="/wnba/prop_picks" element={<LeaguePropPicksPage />} />
        <Route path="/wnba/leaders" element={<LeagueLeadersPage />} />
        <Route path="/wnba/standings" element={<LeagueStandingsPage />} />
        <Route path="/wnba/futures" element={<LeagueFuturesPage />} />
        <Route path="/wnba/player/:playerId" element={<LeaguePlayerPage />} />

        <Route
          path="/nba/matchups"
          element={<LeagueMatchupsPage league="nba" />}
        />
        <Route
          path="/mlb/matchups"
          element={<LeagueMatchupsPage league="mlb" />}
        />
        <Route path="/mlb/games/:gamePk" element={<MlbGameStubPage />} />
      </Route>
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}
