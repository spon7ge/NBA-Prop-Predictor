import { TeamAbbrevAvatar } from "@/components/TeamAbbrevAvatar";
import {
  DEMO_PROP,
  evForSide,
  formatEvPercent,
  type PropSide,
} from "./propExplainerDemo";

export type PropExplainerCardProps = {
  selectedSide: PropSide;
  onSelectSide: (side: PropSide) => void;
};

function formatAmericanOdds(odds: number): string {
  return String(odds).replace("-", "−");
}

export function PropExplainerCard({
  selectedSide,
  onSelectSide,
}: PropExplainerCardProps) {
  const ev = evForSide(selectedSide);
  const evClassName =
    ev > 0
      ? "text-emerald-300 border-emerald-300/30"
      : ev < 0
        ? "text-red-300 border-red-300/30"
        : "text-white/60 border-white/10";

  const sideButtonClass = (side: PropSide) =>
    selectedSide === side
      ? "bg-white text-black"
      : "border border-white/10 bg-white/[0.04] text-white/45";

  return (
    <article className="rounded-2xl border border-white/10 bg-white/[0.04] p-5">
      <header className="mb-4 flex items-center gap-3">
        <TeamAbbrevAvatar
          abbrev={DEMO_PROP.teamAbbrev}
          logoUrl={null}
          sizeClassName="size-10"
        />
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-base font-semibold text-white">
            {DEMO_PROP.playerName}
          </h3>
          <p className="text-sm text-white/50">
            {DEMO_PROP.teamAbbrev} · {DEMO_PROP.position}
          </p>
        </div>
      </header>

      <div className="mb-5 flex items-center justify-between text-xs text-white/45">
        <span>{DEMO_PROP.matchup}</span>
        <span>{DEMO_PROP.tip}</span>
      </div>

      <div className="mb-5 text-center">
        <p className="text-5xl font-semibold tracking-tight text-white">
          {DEMO_PROP.line}
        </p>
        <p className="mt-1 text-sm text-white/50">{DEMO_PROP.stat}</p>
      </div>

      <div className="mb-5 grid grid-cols-3 gap-3 text-center text-xs">
        <div>
          <p className="mb-1 text-white/40">Model</p>
          <p className="text-sm font-medium text-white">{DEMO_PROP.model}</p>
        </div>
        <div>
          <p className="mb-1 text-white/40">EV</p>
          <p
            className={`inline-block rounded-full border px-2 py-0.5 text-sm font-medium ${evClassName}`}
          >
            {formatEvPercent(ev)}
          </p>
        </div>
        <div>
          <p className="mb-1 text-white/40">{DEMO_PROP.bookLabel}</p>
          <p className="text-sm font-medium text-white">
            {formatAmericanOdds(DEMO_PROP.oddsAmerican)}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <button
          type="button"
          aria-pressed={selectedSide === "under"}
          onClick={() => onSelectSide("under")}
          className={`rounded-xl px-4 py-2.5 text-sm font-medium transition-colors ${sideButtonClass("under")}`}
        >
          ↓ Under
        </button>
        <button
          type="button"
          aria-pressed={selectedSide === "over"}
          onClick={() => onSelectSide("over")}
          className={`rounded-xl px-4 py-2.5 text-sm font-medium transition-colors ${sideButtonClass("over")}`}
        >
          ↑ Over
        </button>
      </div>
    </article>
  );
}
