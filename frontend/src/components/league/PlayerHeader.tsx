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

const BIO_ROWS = [
  { key: "height", label: "Height" },
  { key: "birthdate", label: "Birthdate" },
  { key: "college", label: "College" },
  { key: "draft_info", label: "Draft Info" },
] as const;

function buildSubtitle(player: ApiWnbaPlayerResponse): string {
  return [
    player.jersey ? `#${player.jersey}` : null,
    player.position,
    player.team_name,
  ]
    .filter((part): part is string => Boolean(part))
    .join(" · ");
}

export function PlayerHeader({ player }: PlayerHeaderProps) {
  const [imgFailed, setImgFailed] = useState(false);
  const showHeadshot = Boolean(player.headshot_url) && !imgFailed;
  const subtitle = buildSubtitle(player);
  const rows = BIO_ROWS.filter(({ key }) => Boolean(player[key])).map(
    ({ key, label }) => ({ label, value: player[key] as string }),
  );

  return (
    <section className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
      <div className="flex flex-col gap-6 md:flex-row md:items-start md:justify-between">
        <div className="flex min-w-0 flex-1 gap-4">
          {showHeadshot ? (
            <img
              src={player.headshot_url!}
              alt={player.name}
              onError={() => setImgFailed(true)}
              className="size-24 shrink-0 rounded-full object-cover bg-white/5"
            />
          ) : (
            <div
              role="img"
              aria-label={`${player.name} placeholder`}
              className="flex size-24 shrink-0 items-center justify-center rounded-full border border-white/10 bg-white/[0.05] text-sm font-semibold text-white/40"
            >
              {player.team_abbrev.slice(0, 3)}
            </div>
          )}
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-xl font-semibold tracking-tight text-white">
              {player.name}
            </h2>
            {subtitle ? (
              <p className="text-sm text-white/45">{subtitle}</p>
            ) : null}
            {rows.length > 0 ? (
              <dl className="mt-4 space-y-2 text-sm">
                {rows.map(({ label, value }) => (
                  <div key={label} className="grid grid-cols-[7rem_1fr] gap-2">
                    <dt className="text-white/35">{label}</dt>
                    <dd className="text-white">{value}</dd>
                  </div>
                ))}
              </dl>
            ) : null}
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
