import { BarChart3, Settings } from "lucide-react";

const leagues = [
  { id: "nba", label: "NBA", dot: "bg-orange-500" },
  { id: "wnba", label: "WNBA", dot: "bg-orange-400" },
] as const;

export function HomeNav() {
  return (
    <header className="border-b border-white/10 bg-black">
      <div className="mx-auto flex h-12 max-w-6xl items-center justify-between gap-4 px-4 sm:px-6">
        <a href="/" className="flex items-center gap-2 text-white no-underline">
          <BarChart3 className="size-4 shrink-0" aria-hidden />
          <span className="text-sm font-semibold tracking-tight">hoopvista</span>
        </a>

        <nav
          className="hidden items-center gap-5 sm:flex"
          aria-label="Leagues"
        >
          {leagues.map((league) => (
            <a
              key={league.id}
              href="#live-now"
              className="flex items-center gap-2 text-xs font-medium text-white/80 no-underline transition-colors hover:text-white"
            >
              <span
                className={`size-1.5 rounded-full ${league.dot}`}
                aria-hidden
              />
              {league.label}
            </a>
          ))}
        </nav>

        <div className="flex items-center gap-3">
          <a
            href="#"
            className="text-xs font-medium text-white/70 no-underline hover:text-white"
          >
            About
          </a>
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
