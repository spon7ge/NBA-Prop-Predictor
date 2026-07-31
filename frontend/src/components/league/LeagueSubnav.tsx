import { Link, useLocation } from "react-router-dom";
import type { LeagueSlug } from "./types";

type LeagueSubnavProps = {
  league: LeagueSlug;
};

const exploreItems = [
  "Matchups",
  "HoopVista Picks",
  "Leaders",
  "Standings",
  "Playoff race",
  "Clutch",
] as const;

const learnItems = ["How it works", "Glossary"] as const;

export function LeagueSubnav({ league }: LeagueSubnavProps) {
  const { pathname } = useLocation();
  const activeClassName =
    league === "wnba"
      ? "bg-violet-600 text-white"
      : "bg-sky-600 text-white";

  function itemPath(item: string): string | null {
    if (item === "Matchups") return `/${league}/matchups`;
    if (item === "Leaders" && league === "wnba") return "/wnba/leaders";
    if (item === "Standings" && league === "wnba") return "/wnba/standings";
    return null;
  }

  function isActive(item: string): boolean {
    if (item === "Matchups") return pathname.endsWith("/matchups");
    if (item === "Leaders") return pathname.endsWith("/leaders");
    if (item === "Standings") return pathname.endsWith("/standings");
    return false;
  }

  function renderItem(item: string) {
    const href = itemPath(item);
    const active = isActive(item);
    const className = active
      ? `rounded-full px-4 py-2 text-sm font-semibold ${activeClassName}`
      : href
        ? "rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 text-sm font-medium text-white/70 hover:text-white"
        : "cursor-not-allowed rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 text-sm font-medium text-white/35";

    if (href) {
      return (
        <Link
          key={item}
          to={href}
          aria-current={active ? "page" : undefined}
          className={className}
        >
          {item}
        </Link>
      );
    }

    return (
      <button
        key={item}
        type="button"
        disabled
        className={className}
      >
        {item}
      </button>
    );
  }

  return (
    <nav
      aria-label={`${league.toUpperCase()} sections`}
      className="mx-auto max-w-6xl px-4 py-5 sm:px-6"
    >
      <div className="flex gap-6 overflow-x-auto rounded-2xl border border-white/10 bg-[#121212] px-4 py-3">
        <div className="flex shrink-0 items-center gap-2">
          <p className="px-1 text-[10px] font-semibold tracking-[0.18em] text-white/35 uppercase">
            Explore
          </p>
          <div className="flex gap-2">{exploreItems.map(renderItem)}</div>
        </div>
        <div className="flex shrink-0 items-center gap-2 border-l border-white/10 pl-6">
          <p className="px-1 text-[10px] font-semibold tracking-[0.18em] text-white/35 uppercase">
            Learn
          </p>
          <div className="flex gap-2">{learnItems.map(renderItem)}</div>
        </div>
      </div>
    </nav>
  );
}
