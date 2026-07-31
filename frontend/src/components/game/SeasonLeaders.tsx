import type {
  GameDetail,
  GameDetailSeasonLeader,
  GameDetailTeam,
} from "./types";

type SeasonLeadersProps = {
  detail: GameDetail;
};

function LeaderRow({ leader }: { leader: GameDetailSeasonLeader }) {
  return (
    <li className="flex items-baseline gap-3 text-sm">
      <span className="w-16 shrink-0 text-white/45">{leader.label}</span>
      <span className="min-w-0 flex-1 truncate font-medium text-white">
        {leader.name}
      </span>
      <span className="shrink-0 font-medium text-white">{leader.value}</span>
    </li>
  );
}

function LeaderColumn({
  team,
  leaders,
}: {
  team: GameDetailTeam;
  leaders: GameDetailSeasonLeader[];
}) {
  return (
    <div>
      <h3 className="mb-3 flex items-baseline gap-2 text-sm">
        <span className="font-semibold" style={{ color: team.color }}>
          {team.abbrev}
        </span>
        <span className="truncate text-white/45">{team.name}</span>
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

      <div className="mt-4 grid gap-8 md:grid-cols-2">
        <LeaderColumn team={detail.away} leaders={seasonLeaders.away} />
        <LeaderColumn team={detail.home} leaders={seasonLeaders.home} />
      </div>
    </section>
  );
}
