import { Link, useLocation } from "react-router-dom";
import { BarChart3, Settings } from "lucide-react";
import basketball from "@/assets/basketball.png";
import wnbaBasketball from "@/assets/wnba_basketball.png";

const leagues = [
  { id: "nba", label: "NBA", icon: basketball },
  { id: "wnba", label: "WNBA", icon: wnbaBasketball },
] as const;

export function HomeNav() {
  const { pathname } = useLocation();
  const aboutActive = pathname === "/about";

  return (
    <header className="border-b border-white/10 bg-black">
      <div className="mx-auto flex h-12 max-w-6xl items-center justify-between gap-4 px-4 sm:px-6">
        <Link to="/" className="flex items-center gap-2 text-white no-underline">
          <BarChart3 className="size-4 shrink-0" aria-hidden />
          <span className="text-[18px] font-semibold tracking-tight">
            HoopVista
          </span>
        </Link>

        <div className="flex items-center gap-4">
          <nav className="flex items-center gap-3" aria-label="Primary">
            <div className="hidden items-center gap-3 sm:flex">
              {leagues.map((league) => (
                <a
                  key={league.id}
                  href="/#live-now"
                  className="flex items-center gap-2 text-[14px] font-medium text-white/80 no-underline transition-colors hover:text-white"
                >
                  <img
                    src={league.icon}
                    alt=""
                    aria-hidden
                    className="size-4 shrink-0 object-contain"
                  />
                  {league.label}
                </a>
              ))}
            </div>
            <Link
              to="/about"
              aria-current={aboutActive ? "page" : undefined}
              className={
                aboutActive
                  ? "rounded-md bg-neutral-600/80 px-2.5 py-1 text-[14px] font-medium text-white/90 no-underline"
                  : "rounded-md px-2.5 py-1 text-[14px] font-medium text-white/80 no-underline transition-colors hover:bg-neutral-600/50 hover:text-white"
              }
            >
              About
            </Link>
          </nav>

          <button
            type="button"
            aria-label="Settings"
            className="rounded-md p-1.5 text-white/70 transition-colors hover:bg-white/5 hover:text-white"
          >
            <Settings className="size-4" />
          </button>
        </div>
      </div>
    </header>
  );
}
