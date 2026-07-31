import type { ApiWnbaLeaderCategory } from "@/lib/api";
import { LeaderCategoryCard } from "./LeaderCategoryCard";

type LeadersGridProps = {
  season: number;
  categories: ApiWnbaLeaderCategory[];
  isLoading?: boolean;
  isError?: boolean;
};

function Skeletons() {
  return (
    <div
      className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3"
      aria-label="Loading leaders"
    >
      {Array.from({ length: 6 }, (_, i) => (
        <div
          key={i}
          className="h-72 animate-pulse rounded-xl border border-white/10 bg-white/5"
        />
      ))}
    </div>
  );
}

export function LeadersGrid({
  season,
  categories,
  isLoading = false,
  isError = false,
}: LeadersGridProps) {
  return (
    <section className="mx-auto max-w-6xl space-y-4 px-4 sm:px-6">
      <p className="text-sm text-white/45">
        {season} season · per game
      </p>
      {isLoading ? (
        <Skeletons />
      ) : isError ? (
        <p className="text-sm text-white/50">Leaders unavailable</p>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {categories.map((category) => (
            <LeaderCategoryCard key={category.key} category={category} />
          ))}
        </div>
      )}
      <p className="text-xs text-white/35">Data: stats.wnba.com</p>
    </section>
  );
}
