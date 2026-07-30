import type { LeagueSlug } from "./types";

type LeagueSubnavProps = {
  league: LeagueSlug;
};

const groups = [
  {
    label: "Explore",
    items: [
      "Matchups",
      "HoopVista Picks",
      "Leaders",
      "Standings",
      "Playoff race",
      "Clutch",
    ],
  },
  {
    label: "Learn",
    items: ["How it works", "Glossary"],
  },
] as const;

export function LeagueSubnav({ league }: LeagueSubnavProps) {
  const activeClassName =
    league === "wnba"
      ? "bg-violet-600 text-white"
      : "bg-sky-600 text-white";

  return (
    <nav
      aria-label={`${league.toUpperCase()} sections`}
      className="mx-auto max-w-6xl px-4 py-5 sm:px-6"
    >
      <div className="flex gap-6 overflow-x-auto rounded-2xl border border-white/10 bg-[#121212] px-4 py-3">
        {groups.map((group) => (
          <div key={group.label} className="shrink-0">
            <p className="mb-2 px-1 text-[10px] font-semibold tracking-[0.18em] text-white/35 uppercase">
              {group.label}
            </p>
            <div className="flex gap-2">
              {group.items.map((item) => {
                const isActive = item === "Matchups";

                return (
                  <button
                    key={item}
                    type="button"
                    disabled={!isActive}
                    aria-current={isActive ? "page" : undefined}
                    className={
                      isActive
                        ? `rounded-full px-4 py-2 text-sm font-semibold ${activeClassName}`
                        : "cursor-not-allowed rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 text-sm font-medium text-white/35"
                    }
                  >
                    {item}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </nav>
  );
}
