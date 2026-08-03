import { useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { LeagueHero } from "@/components/league/LeagueHero";
import { LeagueSubnav } from "@/components/league/LeagueSubnav";
import { MatchupsPanel } from "@/components/league/MatchupsPanel";
import {
  isOddsWindowDate,
  isValidEtDate,
  parseMatchupDateParam,
  shiftEtDate,
  slateEtDate,
} from "@/components/league/matchupSlateDate";
import { mergeMatchupOdds } from "@/components/league/mergeMatchupOdds";
import type { LeagueSlug } from "@/components/league/types";
import { mapToMatchupGames } from "@/components/home/mapScoreboard";
import { useMlbOdds } from "@/hooks/useMlbOdds";
import { useMlbScoreboard } from "@/hooks/useMlbScoreboard";
import { useWnbaOdds } from "@/hooks/useWnbaOdds";
import { useWnbaScoreboard } from "@/hooks/useWnbaScoreboard";

type LeagueMatchupsPageProps = {
  league: LeagueSlug;
};

export function LeagueMatchupsPage({ league }: LeagueMatchupsPageProps) {
  if (league === "wnba") {
    return <WnbaMatchupsPage />;
  }

  if (league === "mlb") {
    return <MlbMatchupsPage />;
  }

  return (
    <div className="space-y-0 pb-8">
      <LeagueHero league="nba" />
      <LeagueSubnav league="nba" />
      <p className="mx-auto max-w-6xl px-4 text-sm text-white/40 sm:px-6">
        NBA matchups coming soon.
      </p>
    </div>
  );
}

function WnbaMatchupsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const today = slateEtDate();
  const raw = searchParams.get("date");
  const selectedDate = parseMatchupDateParam(raw, today);
  const showOdds = isOddsWindowDate(selectedDate, today);

  useEffect(() => {
    if (raw !== null && !isValidEtDate(raw)) {
      setSearchParams({}, { replace: true });
    }
  }, [raw, setSearchParams]);

  const { games, isLoading, hasNeverLoaded, data } =
    useWnbaScoreboard(selectedDate);
  const oddsQuery = useWnbaOdds();
  const matchupGames = mergeMatchupOdds(
    mapToMatchupGames(games),
    showOdds ? oddsQuery.data?.games : undefined,
    selectedDate,
  );

  const setDate = (next: string) => {
    if (next === today) setSearchParams({});
    else setSearchParams({ date: next });
  };

  return (
    <div className="space-y-0">
      <LeagueHero league="wnba" dateEt={data?.date ?? selectedDate} />
      <LeagueSubnav league="wnba" />
      <MatchupsPanel
        games={matchupGames}
        isLoading={isLoading}
        isError={hasNeverLoaded}
        selectedDate={selectedDate}
        todayDate={today}
        onPrevDay={() => setDate(shiftEtDate(selectedDate, -1))}
        onNextDay={() => setDate(shiftEtDate(selectedDate, 1))}
        onGoToday={() => setDate(today)}
      />
    </div>
  );
}

function MlbMatchupsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const today = slateEtDate();
  const raw = searchParams.get("date");
  const selectedDate = parseMatchupDateParam(raw, today);
  const showOdds = isOddsWindowDate(selectedDate, today);

  useEffect(() => {
    if (raw !== null && !isValidEtDate(raw)) {
      setSearchParams({}, { replace: true });
    }
  }, [raw, setSearchParams]);

  const { games, isLoading, hasNeverLoaded, data } =
    useMlbScoreboard(selectedDate);
  const oddsQuery = useMlbOdds();
  const matchupGames = mergeMatchupOdds(
    mapToMatchupGames(games),
    showOdds ? oddsQuery.data?.games : undefined,
    selectedDate,
    { wnbaAliases: false },
  );

  const setDate = (next: string) => {
    if (next === today) setSearchParams({});
    else setSearchParams({ date: next });
  };

  return (
    <div className="space-y-0">
      <LeagueHero league="mlb" dateEt={data?.date ?? selectedDate} />
      <LeagueSubnav league="mlb" />
      <MatchupsPanel
        games={matchupGames}
        isLoading={isLoading}
        isError={hasNeverLoaded}
        selectedDate={selectedDate}
        todayDate={today}
        onPrevDay={() => setDate(shiftEtDate(selectedDate, -1))}
        onNextDay={() => setDate(shiftEtDate(selectedDate, 1))}
        onGoToday={() => setDate(today)}
      />
    </div>
  );
}
