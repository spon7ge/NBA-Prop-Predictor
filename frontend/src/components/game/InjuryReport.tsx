import { GameSection } from "./GameSection";
import type { GameDetail, GameDetailInjury } from "./types";

type InjuryReportProps = {
  detail: GameDetail;
};

function InjuryRow({ injury }: { injury: GameDetailInjury }) {
  return (
    <li className="text-sm text-white/80">
      <div className="flex items-baseline gap-1.5">
        <span>{injury.name}</span>
        {injury.position ? (
          <span className="text-white/50">{injury.position}</span>
        ) : null}
      </div>
      <div className="mt-0.5 text-xs">
        <span className="font-medium text-white">{injury.status}</span>
        {injury.detail ? (
          <span className="text-white/50">{` · ${injury.detail}`}</span>
        ) : null}
      </div>
    </li>
  );
}

function InjuryColumn({
  abbrev,
  color,
  injuries,
  showEmptyPlaceholder,
}: {
  abbrev: string;
  color: string;
  injuries: GameDetailInjury[];
  showEmptyPlaceholder: boolean;
}) {
  return (
    <div>
      <h3
        className="mb-2 text-xs font-semibold uppercase tracking-wide"
        style={{ color }}
      >
        {abbrev}
      </h3>
      {injuries.length > 0 ? (
        <ul className="space-y-2">
          {injuries.map((injury) => (
            <InjuryRow key={injury.name} injury={injury} />
          ))}
        </ul>
      ) : showEmptyPlaceholder ? (
        <p className="text-sm text-white/40">None listed</p>
      ) : null}
    </div>
  );
}

export function InjuryReport({ detail }: InjuryReportProps) {
  const injuries = detail.injuries;

  if (!injuries) {
    return null;
  }

  const awayHasInjuries = injuries.away.length > 0;
  const homeHasInjuries = injuries.home.length > 0;
  const eitherSideHasInjuries = awayHasInjuries || homeHasInjuries;

  return (
    <GameSection>
      <h2 className="text-sm font-semibold text-white">Injury report</h2>

      <div className="mt-4 grid gap-6 md:grid-cols-2">
        <InjuryColumn
          abbrev={detail.away.abbrev}
          color={detail.away.color}
          injuries={injuries.away}
          showEmptyPlaceholder={eitherSideHasInjuries && !awayHasInjuries}
        />
        <InjuryColumn
          abbrev={detail.home.abbrev}
          color={detail.home.color}
          injuries={injuries.home}
          showEmptyPlaceholder={eitherSideHasInjuries && !homeHasInjuries}
        />
      </div>
    </GameSection>
  );
}
