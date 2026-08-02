import { Link, useLocation } from "react-router-dom";
import { BarChart3, Settings } from "lucide-react";
import nbaLogo from "@/assets/nba_logo.png";
import wnbaLogo from "@/assets/wnba_logo.png";

const MLB_LOGO =
  "https://a.espncdn.com/i/teamlogos/leagues/500/mlb.png";

const leagues = [
  { id: "nba", label: "NBA", icon: nbaLogo },
  { id: "wnba", label: "WNBA", icon: wnbaLogo },
  { id: "mlb", label: "MLB", icon: MLB_LOGO },
] as const;

export function HomeNav() {
  const { pathname } = useLocation();
  const aboutActive = pathname === "/about";

  return (
    <header className="border-b border-white/10 bg-black">
      <div className="mx-auto flex h-12 max-w-6xl items-center justify-between gap-4 px-4 sm:px-6">
        <Link to="/" className="flex items-center gap-2 text-white no-underline">
          <BarChart3 className="size-4 shrink-0" aria-hidden strokeWidth={1.75} />
          <span className="text-[17px] font-semibold tracking-tight">
            HoopVista
          </span>
        </Link>

        <div className="flex items-center gap-4">
          <nav className="flex items-center gap-3" aria-label="Primary">
            <div className="hidden items-center gap-1 sm:flex">
              {leagues.map((league) => {
                const active = pathname.startsWith(`/${league.id}`);

                return (
                  <Link
                    key={league.id}
                    to={`/${league.id}/matchups`}
                    aria-current={active ? "page" : undefined}
                    className={
                      active
                        ? "flex items-center gap-2 rounded-md bg-white/10 px-3 py-1 text-[14px] font-medium text-white no-underline"
                        : "flex items-center gap-2 rounded-md px-3 py-1 text-[14px] font-medium text-white/55 no-underline transition-colors hover:text-white"
                    }
                  >
                    <img
                      src={league.icon}
                      alt=""
                      aria-hidden
                      className="size-5 shrink-0 object-contain"
                    />
                    {league.label}
                  </Link>
                );
              })}
            </div>
            <Link
              to="/about"
              aria-current={aboutActive ? "page" : undefined}
              className={
                aboutActive
                  ? "rounded-md bg-white/10 px-2.5 py-1 text-[14px] font-medium text-white no-underline"
                  : "rounded-md px-2.5 py-1 text-[14px] font-medium text-white/55 no-underline transition-colors hover:bg-white/5 hover:text-white"
              }
            >
              About
            </Link>
          </nav>

          <button
            type="button"
            aria-label="Settings"
            className="rounded-md p-1.5 text-white/45 transition-colors hover:bg-white/5 hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
          >
            <Settings className="size-4" strokeWidth={1.75} />
          </button>
        </div>
      </div>
    </header>
  );
}
