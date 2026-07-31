import type { GameDetail, GameDetailStarter, GameDetailTeam } from "./types";

type ProjectedStartersProps = {
  detail: GameDetail;
};

function StarterRow({ starter }: { starter: GameDetailStarter }) {
  return (
    <li className="flex items-baseline justify-between gap-3 text-sm">
      <span className="min-w-0 truncate">
        {starter.jersey ? (
          <span className="text-white/45">#{starter.jersey}</span>
        ) : null}
        {starter.jersey ? " " : null}
        <span className="font-medium text-white">{starter.name}</span>
      </span>
      {starter.position ? (
        <span className="shrink-0 text-white/45">{starter.position}</span>
      ) : null}
    </li>
  );
}

function StarterColumn({
  team,
  starters,
}: {
  team: GameDetailTeam;
  starters: GameDetailStarter[];
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
        {starters.map((starter) => (
          <StarterRow
            key={`${starter.jersey ?? "na"}-${starter.name}`}
            starter={starter}
          />
        ))}
      </ul>
    </div>
  );
}

export function ProjectedStarters({ detail }: ProjectedStartersProps) {
  const projectedStarters = detail.projectedStarters;

  if (!projectedStarters) {
    return null;
  }

  return (
    <section className="rounded-xl border border-white/10 bg-[#141414] p-4">
      <h2 className="text-sm font-semibold text-white">
        Projected starters
        <span className="font-normal text-white/45">
          {" "}
          · {projectedStarters.note}
        </span>
      </h2>

      <div className="mt-4 grid gap-8 md:grid-cols-2">
        <StarterColumn team={detail.away} starters={projectedStarters.away} />
        <StarterColumn team={detail.home} starters={projectedStarters.home} />
      </div>
    </section>
  );
}
