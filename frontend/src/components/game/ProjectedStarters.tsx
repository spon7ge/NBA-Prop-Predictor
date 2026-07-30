import type { GameDetail, GameDetailStarter } from "./types";

type ProjectedStartersProps = {
  detail: GameDetail;
};

function StarterRow({ starter }: { starter: GameDetailStarter }) {
  return (
    <li className="text-sm text-white/80">
      {starter.jersey ? `#${starter.jersey} ` : null}
      <span>{starter.name}</span>
      {starter.position ? ` ${starter.position}` : null}
    </li>
  );
}

function StarterColumn({
  abbrev,
  color,
  starters,
}: {
  abbrev: string;
  color: string;
  starters: GameDetailStarter[];
}) {
  return (
    <div>
      <h3
        className="mb-2 text-xs font-semibold uppercase tracking-wide"
        style={{ color }}
      >
        {abbrev}
      </h3>
      <ul className="space-y-1.5">
        {starters.map((starter) => (
          <StarterRow key={`${starter.jersey ?? "na"}-${starter.name}`} starter={starter} />
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
      <h2 className="text-sm font-semibold text-white">Projected starters</h2>
      <p className="mt-1 text-xs text-white/50">{projectedStarters.note}</p>

      <div className="mt-4 grid gap-6 md:grid-cols-2">
        <StarterColumn
          abbrev={detail.away.abbrev}
          color={detail.away.color}
          starters={projectedStarters.away}
        />
        <StarterColumn
          abbrev={detail.home.abbrev}
          color={detail.home.color}
          starters={projectedStarters.home}
        />
      </div>
    </section>
  );
}
