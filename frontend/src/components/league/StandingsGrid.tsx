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
          className="h-72 animate-pulse rounded-xl border border-white/10 bg-white/5"
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
    <section className="mx-auto max-w-6xl space-y-4 px-4 sm:px-6">
      <p className="text-sm text-white/45">{season} regular season</p>
      {isLoading ? (
        <Skeletons />
      ) : isError ? (
        <p className="text-sm text-white/50">Standings unavailable</p>
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
