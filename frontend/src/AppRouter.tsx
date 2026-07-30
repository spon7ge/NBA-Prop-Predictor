import { Routes, Route } from "react-router-dom";
import { HomeChromeLayout } from "@/layouts/HomeChromeLayout";
import { HomePage } from "@/pages/HomePage";
import { AboutPage } from "@/pages/AboutPage";
import { GameDetailPage } from "@/pages/GameDetailPage";
import { LeagueMatchupsPage } from "@/pages/LeagueMatchupsPage";
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
        <Route
          path="/nba/matchups"
          element={<LeagueMatchupsPage league="nba" />}
        />
      </Route>
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}
