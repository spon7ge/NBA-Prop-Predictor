import type { ApiWnbaStandingsConference } from "@/lib/api";
import { TeamAbbrevAvatar } from "@/components/TeamAbbrevAvatar";
import { teamColor } from "./wnbaTeamColors";

type StandingsConferenceCardProps = {
  conference: ApiWnbaStandingsConference;
};

function diffClass(diff: string): string {
  if (diff.startsWith("+")) return "text-white/80";
  if (diff.startsWith("-") && diff !== "-") return "text-white/45";
  return "text-white/40";
}

function streakClass(streak: string): string {
  if (streak.startsWith("W")) return "text-white/80";
  if (streak.startsWith("L")) return "text-white/45";
  return "text-white/40";
}

export function StandingsConferenceCard({
  conference,
}: StandingsConferenceCardProps) {
  return (
    <section className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
      <h3 className="mb-3 text-base font-semibold tracking-tight text-white">
        {conference.label}
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="text-[10px] tracking-wide text-white/40 uppercase">
              <th className="pb-2 pr-1.5 font-medium">#</th>
              <th className="pb-2 pr-1 font-medium">Team</th>
              <th className="pb-2 pr-1.5 font-medium">W-L</th>
              <th className="pb-2 pr-1.5 font-medium">PCT</th>
              <th className="pb-2 pr-1.5 font-medium">GB</th>
              <th className="pb-2 pr-1.5 font-medium">Home</th>
              <th className="pb-2 pr-1.5 font-medium">Away</th>
              <th className="pb-2 pr-1.5 font-medium">L10</th>
              <th className="pb-2 pr-1.5 font-medium">Diff</th>
              <th className="pb-2 font-medium">Strk</th>
            </tr>
          </thead>
          <tbody>
            {conference.teams.length === 0 ? (
              <tr>
                <td colSpan={10} className="py-3 text-white/40">
                  No data
                </td>
              </tr>
            ) : (
              conference.teams.map((row) => (
                <tr key={`${conference.key}-${row.team_id}`}>
                  <td className="py-1 pr-1.5 text-white/50">{row.rank}</td>
                  <td className="py-1 pr-1">
                    <div className="flex items-center gap-1.5">
                      <TeamAbbrevAvatar
                        abbrev={row.abbrev}
                        logoUrl={row.logo_url}
                        sizeClassName="size-5"
                      />
                      <span
                        className="font-semibold"
                        style={{ color: teamColor(row.abbrev) }}
                      >
                        {row.abbrev}
                      </span>
                      <span className="text-white/80">{row.name}</span>
                    </div>
                  </td>
                  <td className="py-1 pr-1.5 tabular-nums text-white">{row.wl}</td>
                  <td className="py-1 pr-1.5 tabular-nums text-white/70">{row.pct}</td>
                  <td className="py-1 pr-1.5 tabular-nums text-white/70">{row.gb}</td>
                  <td className="py-1 pr-1.5 tabular-nums text-white/70">{row.home}</td>
                  <td className="py-1 pr-1.5 tabular-nums text-white/70">{row.away}</td>
                  <td className="py-1 pr-1.5 tabular-nums text-white/70">{row.l10}</td>
                  <td
                    className={`py-1 pr-1.5 font-medium tabular-nums ${diffClass(row.diff)}`}
                  >
                    {row.diff}
                  </td>
                  <td className={`py-1 font-medium tabular-nums ${streakClass(row.streak)}`}>
                    {row.streak}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
