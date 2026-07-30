import { Link } from "react-router-dom";
import type { GameDetail, GameDetailTeam } from "./types";

const statusAccent: Record<GameDetail["status"], string> = {
  scheduled: "text-white/55",
  live: "text-red-400",
  halftime: "text-red-400",
  final: "text-white/55",
};

function ScoreBox({ score }: { score: number | null }) {
  return (
    <span className="flex size-12 shrink-0 items-center justify-center rounded-md bg-black font-mono text-xl font-bold text-amber-300">
      {score ?? "–"}
    </span>
  );
}

function TeamRow({ team }: { team: GameDetailTeam }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-base font-semibold" style={{ color: team.color }}>
        {team.name}
      </span>
      <ScoreBox score={team.score} />
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

      <div className="rounded-xl border border-white/10 bg-[#141414] p-4">
        <p className="mb-3 flex items-center gap-2 text-[14px] text-white/55">
          {inProgress ? (
            <span
              className="size-1.5 shrink-0 rounded-full bg-violet-500"
              aria-hidden
            />
          ) : (
            <span
              className="size-1.5 shrink-0 rounded-full bg-white/25"
              aria-hidden
            />
          )}
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
