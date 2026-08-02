import { useState } from "react";
import type { ApiWnbaPlayerResponse } from "@/lib/api";

type PlayerHeaderProps = {
  player: ApiWnbaPlayerResponse;
};

const AVG_TILES = [
  { key: "pts", label: "PTS" },
  { key: "reb", label: "REB" },
  { key: "ast", label: "AST" },
  { key: "fg_pct", label: "FG%" },
  { key: "fg3_pct", label: "3P%" },
] as const;

export function PlayerHeader({ player }: PlayerHeaderProps) {
  const [imgFailed, setImgFailed] = useState(false);
  const showHeadshot = Boolean(player.headshot_url) && !imgFailed;

  return (
    <section className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
      <div className="flex flex-col gap-5 md:flex-row md:items-center md:justify-between">
        <div className="flex items-center gap-3">
          {showHeadshot ? (
            <img
              src={player.headshot_url!}
              alt={player.name}
              onError={() => setImgFailed(true)}
              className="size-16 shrink-0 rounded-full object-cover bg-white/5"
            />
          ) : (
            <div
              role="img"
              aria-label={`${player.name} placeholder`}
              className="flex size-16 shrink-0 items-center justify-center rounded-full border border-white/10 bg-white/[0.05] text-sm font-semibold text-white/40"
            >
              {player.team_abbrev.slice(0, 3)}
            </div>
          )}
          <div className="min-w-0">
            <h2 className="truncate text-xl font-semibold tracking-tight text-white">
              {player.name}
            </h2>
            <p className="mt-0.5 text-sm text-white/45">
              {player.position ? (
                <>
                  <span>{player.position}</span>
                  <span className="mx-1.5 text-white/25">·</span>
                </>
              ) : null}
              <span>{player.team_name}</span>
            </p>
          </div>
        </div>

        <div className="grid grid-cols-5 gap-2 sm:gap-3">
          {AVG_TILES.map(({ key, label }) => (
            <div
              key={key}
              className="rounded-lg border border-white/10 bg-white/[0.03] px-2 py-2 text-center"
            >
              <div className="text-[10px] font-medium tracking-wide text-white/35 uppercase">
                {label}
              </div>
              <div className="mt-1 text-base font-semibold tabular-nums text-white">
                {player.averages[key]}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
