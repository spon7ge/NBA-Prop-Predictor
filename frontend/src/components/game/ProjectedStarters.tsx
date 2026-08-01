import { GameSection } from "./GameSection";
import type { GameDetail, GameDetailStarter, GameDetailTeam } from "./types";

type ProjectedStartersProps = {
  detail: GameDetail;
};

function StarterRow({ starter }: { starter: GameDetailStarter }) {
  return (
    <li className="flex items-baseline justify-between gap-3 text-sm">
      <span className="flex min-w-0 items-baseline gap-1.5 truncate">
        {starter.jersey ? (
          <span className="shrink-0 text-white/45">#{starter.jersey}</span>
        ) : null}
        <span className="truncate font-medium text-white">{starter.name}</span>
        {starter.gtd ? (
          <span className="shrink-0 rounded-full bg-red-600 px-1.5 py-0.5 text-[10px] font-semibold uppercase leading-none tracking-wide text-white">
            GTD
          </span>
        ) : null}
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
    <GameSection>
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
    </GameSection>
  );
}
