import { Link, useLocation } from "react-router-dom";
import type { LeagueSlug } from "./types";

type LeagueSubnavProps = {
  league: LeagueSlug;
};

const exploreItems = [
  "Matchups",
  "Prop Picks",
  "Leaders",
  "Standings",
  "Playoff race",
  "Futures",
] as const;

const learnItems = ["How it works", "Glossary"] as const;

export function LeagueSubnav({ league }: LeagueSubnavProps) {
  const { pathname } = useLocation();

  function itemPath(item: string): string | null {
    if (item === "Matchups") return `/${league}/matchups`;
    if (item === "Prop Picks" && league === "wnba") return "/wnba/prop_picks";
    if (item === "Leaders" && league === "wnba") return "/wnba/leaders";
    if (item === "Standings" && league === "wnba") return "/wnba/standings";
    if (item === "Futures" && league === "wnba") return "/wnba/futures";
    return null;
  }

  function isActive(item: string): boolean {
    if (item === "Matchups") return pathname.endsWith("/matchups");
    if (item === "Prop Picks") return pathname.endsWith("/prop_picks");
    if (item === "Leaders") return pathname.endsWith("/leaders");
    if (item === "Standings") return pathname.endsWith("/standings");
    if (item === "Futures") return pathname.endsWith("/futures");
    return false;
  }

  function renderItem(item: string) {
    const href = itemPath(item);
    const active = isActive(item);
    const className = active
      ? "rounded-md bg-white/10 px-3 py-1.5 text-sm font-medium text-white"
      : href
        ? "rounded-md px-3 py-1.5 text-sm font-medium text-white/55 transition-colors hover:text-white"
        : "cursor-not-allowed rounded-md px-3 py-1.5 text-sm font-medium text-white/25";

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
      <button key={item} type="button" disabled className={className}>
        {item}
      </button>
    );
  }

  return (
    <nav
      aria-label={`${league.toUpperCase()} sections`}
      className="mx-auto max-w-6xl px-4 py-6 sm:px-6"
    >
      <div className="flex gap-6 overflow-x-auto border-b border-white/10 pb-4">
        <div className="flex shrink-0 items-center gap-2">
          <p className="px-1 text-[10px] font-medium tracking-[0.18em] text-white/35 uppercase">
            Explore
          </p>
          <div className="flex gap-1">{exploreItems.map(renderItem)}</div>
        </div>
        <div className="flex shrink-0 items-center gap-2 border-l border-white/10 pl-6">
          <p className="px-1 text-[10px] font-medium tracking-[0.18em] text-white/35 uppercase">
            Learn
          </p>
          <div className="flex gap-1">{learnItems.map(renderItem)}</div>
        </div>
      </div>
    </nav>
  );
}
