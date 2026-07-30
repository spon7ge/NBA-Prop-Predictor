import type { GameDetail, GameDetailSeasonLeader } from "./types";

type SeasonLeadersProps = {
  detail: GameDetail;
};

function LeaderRow({ leader }: { leader: GameDetailSeasonLeader }) {
  return (
    <li className="text-sm text-white/80">
      <span className="text-white/50">{leader.label}</span>
      <div className="mt-0.5 flex items-baseline justify-between gap-2">
        <span>{leader.name}</span>
        <span className="font-medium text-white">{leader.value}</span>
      </div>
    </li>
  );
}

function LeaderColumn({
  abbrev,
  color,
  leaders,
}: {
  abbrev: string;
  color: string;
  leaders: GameDetailSeasonLeader[];
}) {
  return (
    <div>
      <h3
        className="mb-2 text-xs font-semibold uppercase tracking-wide"
        style={{ color }}
      >
        {abbrev}
      </h3>
      <ul className="space-y-2">
        {leaders.map((leader) => (
          <LeaderRow key={leader.stat} leader={leader} />
        ))}
      </ul>
    </div>
  );
}

export function SeasonLeaders({ detail }: SeasonLeadersProps) {
  const seasonLeaders = detail.seasonLeaders;

  if (!seasonLeaders) {
    return null;
  }

  return (
    <section className="rounded-xl border border-white/10 bg-[#141414] p-4">
      <h2 className="text-sm font-semibold text-white">Season leaders</h2>

      <div className="mt-4 grid gap-6 md:grid-cols-2">
        <LeaderColumn
          abbrev={detail.away.abbrev}
          color={detail.away.color}
          leaders={seasonLeaders.away}
        />
        <LeaderColumn
          abbrev={detail.home.abbrev}
          color={detail.home.color}
          leaders={seasonLeaders.home}
        />
      </div>
    </section>
  );
}
