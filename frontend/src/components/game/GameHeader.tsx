import { useState } from "react";
import { Link } from "react-router-dom";
import { GAME_SECTION_SURFACE } from "./GameSection";
import type { GameDetail, GameDetailTeam } from "./types";

const statusAccent: Record<GameDetail["status"], string> = {
  scheduled: "text-white/55",
  live: "text-red-400",
  halftime: "text-red-400",
  final: "text-white/55",
};

function TeamLogo({ url }: { url: string }) {
  const [failed, setFailed] = useState(false);
  if (failed) return null;
  return (
    <img
      src={url}
      alt=""
      role="presentation"
      className="size-8 shrink-0 object-contain"
      onError={() => setFailed(true)}
    />
  );
}

function ScoreValue({ score }: { score: number | null }) {
  return (
    <span className="shrink-0 font-mono text-xl font-semibold tracking-tight text-white">
      {score ?? "–"}
    </span>
  );
}

function TeamRow({ team }: { team: GameDetailTeam }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="flex min-w-0 items-center gap-2.5">
        {team.logoUrl ? <TeamLogo url={team.logoUrl} /> : null}
        <span
          className="truncate text-base font-semibold"
          style={{ color: team.color }}
        >
          {team.name}
        </span>
      </span>
      <ScoreValue score={team.score} />
    </div>
  );
}

export function GameHeader({ detail }: { detail: GameDetail }) {
  const inProgress = detail.status === "live" || detail.status === "halftime";

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <Link
          to="/"
          className="flex items-center gap-1 text-sm font-medium text-white/70 no-underline transition-colors hover:text-white"
        >
          ← Back
        </Link>
        <span
          className={`flex items-center gap-2 text-xs font-medium ${statusAccent[detail.status]}`}
        >
          {inProgress ? (
            <span
              className="size-1.5 animate-pulse rounded-full bg-red-500"
              aria-hidden
            />
          ) : null}
          {detail.statusLabel}
        </span>
      </div>

      <div className={GAME_SECTION_SURFACE}>
        <p className="mb-3 flex items-center gap-2 text-[14px] text-white/55">
          <span
            className="size-1.5 shrink-0 rounded-full bg-white/25"
            aria-hidden
          />
          <span>
            <span className="text-white/80">{detail.statusLabel}</span>
            {detail.venue ? (
              <>
                <span className="mx-1.5 text-white/30" aria-hidden>
                  ·
                </span>
                <span>{detail.venue}</span>
              </>
            ) : null}
          </span>
        </p>
        <div className="space-y-3">
          <TeamRow team={detail.away} />
          <TeamRow team={detail.home} />
        </div>
      </div>
    </div>
  );
}
