import { ChevronRight } from "lucide-react";
import { Link } from "react-router-dom";
import draftKingsLogo from "@/assets/draftkings.png";
import { isInProgressStatus } from "@/components/home/mapScoreboard";
import { TeamAbbrevAvatar } from "@/components/TeamAbbrevAvatar";
import { formatOddsPill } from "./mergeMatchupOdds";
import type { MatchupGame, MatchupTeam } from "./types";

function TeamRow({
  team,
  showScore,
}: {
  team: MatchupTeam;
  showScore: boolean;
}) {
  return (
    <div className="flex items-center gap-2.5">
      <TeamAbbrevAvatar
        abbrev={team.abbrev}
        logoUrl={team.logoUrl}
        sizeClassName="size-8"
      />
      <span className="w-9 shrink-0 text-xs font-semibold text-white">
        {team.abbrev}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-xs text-white">{team.name}</span>
        {team.record ? (
          <span className="block text-[11px] text-white/45">{team.record}</span>
        ) : null}
      </span>
      {showScore ? (
        <span className="flex size-9 shrink-0 items-center justify-center rounded-md bg-black font-mono text-sm font-bold text-amber-300">
          {team.score ?? "–"}
        </span>
      ) : null}
    </div>
  );
}

export function MatchupGameCard({ game }: { game: MatchupGame }) {
  const isLive = isInProgressStatus(game.status);
  const showScores = game.status !== "scheduled";
  const venueLabel = game.venue
    ? [game.venue, game.venueCity].filter(Boolean).join(" · ")
    : null;
  const accentClassName =
    game.league === "wnba"
      ? isLive
        ? "border-l-violet-500"
        : "border-l-violet-400/80"
      : isLive
        ? "border-l-sky-500"
        : "border-l-sky-400/80";
  const baseClassName = `block rounded-xl border border-l-2 border-white/10 bg-[#141414] p-4 ${accentClassName}`;
  const oddsLabel = game.odds ? formatOddsPill(game.odds) : null;

  const content = (
    <div className="flex items-center gap-3">
      <div className="min-w-0 flex-1">
        <div className="mb-4 flex items-start justify-between gap-3">
          <span
            className={`flex shrink-0 items-center gap-2 text-xs ${
              isLive ? "text-violet-300" : "text-white/55"
            }`}
          >
            {isLive ? (
              <span className="size-1.5 animate-pulse rounded-full bg-violet-500" />
            ) : null}
            {game.statusLabel}
          </span>
          {venueLabel ? (
            <span className="truncate text-right text-[11px] text-white/40">
              {venueLabel}
            </span>
          ) : null}
        </div>
        <div className="space-y-3">
          <TeamRow team={game.away} showScore={showScores} />
          <div className="flex items-center gap-3">
            <div className="min-w-0 flex-1">
              <TeamRow team={game.home} showScore={showScores} />
            </div>
            {oddsLabel ? (
              <div className="shrink-0 text-right">
                <span className="inline-flex max-w-[11rem] rounded-full border border-white/10 bg-white/5 px-2.5 py-1 font-mono text-[11px] text-white/75 sm:max-w-none">
                  {oddsLabel}
                </span>
                <p className="mt-1 flex items-center justify-end gap-1 text-[10px] tracking-wide text-white/35">
                  <span>Odds by</span>
                  <img
                    src={draftKingsLogo}
                    alt="DraftKings"
                    className="h-4 w-4 object-contain"
                  />
                </p>
              </div>
            ) : null}
          </div>
        </div>
      </div>
      <ChevronRight
        aria-hidden="true"
        className="size-4 shrink-0 text-white/25"
      />
    </div>
  );

  if (game.espnEventId) {
    return (
      <Link
        to={`/games/${game.espnEventId}`}
        className={`${baseClassName} transition-colors hover:border-white/20`}
      >
        {content}
      </Link>
    );
  }

  return <article className={baseClassName}>{content}</article>;
}
