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
          <span className="block text-[11px] text-white/40">{team.record}</span>
        ) : null}
      </span>
      {showScore ? (
        <span className="shrink-0 font-mono text-sm font-semibold tracking-tight text-white">
          {team.score ?? "–"}
        </span>
      ) : null}
    </div>
  );
}

function OddsBlock({
  label,
  placement,
}: {
  label: string;
  placement: "under-scores" | "beside-home";
}) {
  return (
    <div
      data-testid="matchup-odds"
      data-placement={placement}
      className="shrink-0 text-right"
    >
      <span className="inline-flex max-w-[11rem] rounded-full border border-white/10 bg-white/[0.03] px-2.5 py-1 font-mono text-[11px] text-white/70 sm:max-w-none">
        {label}
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
  );
}

export function MatchupGameCard({ game }: { game: MatchupGame }) {
  const isLive = isInProgressStatus(game.status);
  const showScores = game.status !== "scheduled";
  const venueLabel = game.venue
    ? [game.venue, game.venueCity].filter(Boolean).join(" · ")
    : null;
  const baseClassName =
    "block rounded-xl border border-white/10 bg-white/[0.03] p-4";
  const oddsLabel = game.odds ? formatOddsPill(game.odds) : null;

  const content = (
    <div className="flex items-center gap-3">
      <div className="min-w-0 flex-1">
        <div className="mb-4 flex items-start justify-between gap-3">
          <span
            className={`flex shrink-0 items-center gap-2 text-xs ${
              isLive ? "text-red-400" : "text-white/45"
            }`}
          >
            {isLive ? (
              <span className="size-1.5 animate-pulse rounded-full bg-red-500" />
            ) : null}
            {game.statusLabel}
          </span>
          {venueLabel ? (
            <span className="truncate text-right text-[11px] text-white/35">
              {venueLabel}
            </span>
          ) : null}
        </div>
        {showScores ? (
          <div className="space-y-3">
            <TeamRow team={game.away} showScore />
            <TeamRow team={game.home} showScore />
            {oddsLabel ? (
              <div className="flex justify-end">
                <OddsBlock label={oddsLabel} placement="under-scores" />
              </div>
            ) : null}
          </div>
        ) : (
          <div className="space-y-3">
            <TeamRow team={game.away} showScore={false} />
            <div className="flex items-center gap-3">
              <div className="min-w-0 flex-1">
                <TeamRow team={game.home} showScore={false} />
              </div>
              {oddsLabel ? (
                <OddsBlock label={oddsLabel} placement="beside-home" />
              ) : null}
            </div>
          </div>
        )}
      </div>
      <ChevronRight
        aria-hidden="true"
        className="size-4 shrink-0 text-white/25"
        strokeWidth={1.75}
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
