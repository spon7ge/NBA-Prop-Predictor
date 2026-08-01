import type { ApiWnbaStandingsConference } from "@/lib/api";
import { StandingsConferenceCard } from "./StandingsConferenceCard";

type StandingsGridProps = {
  season: number;
  conferences: ApiWnbaStandingsConference[];
  isLoading?: boolean;
  isError?: boolean;
};

function Skeletons() {
  return (
    <div
      className="grid grid-cols-1 gap-4 lg:grid-cols-2"
      aria-label="Loading standings"
    >
      {Array.from({ length: 2 }, (_, i) => (
        <div
          key={i}
          className="h-72 animate-pulse rounded-xl border border-white/10 bg-white/[0.03]"
        />
      ))}
    </div>
  );
}

export function StandingsGrid({
  season,
  conferences,
  isLoading = false,
  isError = false,
}: StandingsGridProps) {
  return (
    <section className="mx-auto max-w-6xl space-y-6 px-4 pb-16 sm:px-6 sm:pb-20">
      <header>
        <h2 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">
          Standings
        </h2>
        <p className="mt-2 text-sm text-white/40">{season} regular season</p>
      </header>
      {isLoading ? (
        <Skeletons />
      ) : isError ? (
        <p className="text-sm text-white/40">Standings unavailable</p>
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {conferences.map((conference) => (
            <StandingsConferenceCard
              key={conference.key}
              conference={conference}
            />
          ))}
        </div>
      )}
      <p className="text-xs text-white/35">Data: ESPN</p>
    </section>
  );
}
