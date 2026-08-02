import { TeamAbbrevAvatar } from "@/components/TeamAbbrevAvatar";
import { DEMO_PROP, formatEvPercent } from "./propExplainerDemo";

function formatAmericanOdds(odds: number): string {
  return String(odds).replace("-", "−");
}

export function PropExplainerCard() {
  const ev = DEMO_PROP.ev;
  const evClassName =
    ev > 0
      ? "text-emerald-300 border-emerald-300/30"
      : ev < 0
        ? "text-red-300 border-red-300/30"
        : "text-white/60 border-white/10";

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

      <div
        className="rounded-xl bg-white px-4 py-2.5 text-center text-sm font-medium text-black"
        aria-label={`Example side ${DEMO_PROP.side}`}
      >
        ↑ {DEMO_PROP.side}
      </div>
    </article>
  );
}
