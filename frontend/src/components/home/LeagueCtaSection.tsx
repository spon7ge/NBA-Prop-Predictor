import { Link } from "react-router-dom";

const LEAGUES = [
  { id: "nba", label: "NBA", href: "/nba/matchups" },
  { id: "wnba", label: "WNBA", href: "/wnba/matchups" },
  { id: "mlb", label: "MLB", href: "/mlb/matchups" },
] as const;

export function LeagueCtaSection() {
  return (
    <section
      id="enter-a-league"
      className="mx-auto max-w-6xl border-t border-white/10 px-4 py-16 sm:px-6 sm:py-20"
    >
      <div className="mx-auto max-w-3xl text-center">
        <h2 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">
          Enter a league
        </h2>
        <p className="mt-3 text-sm text-white/40 sm:text-base">
          Matchups, props, and standings — pick your court.
        </p>
      </div>
      <div className="mx-auto mt-10 grid max-w-3xl gap-4 sm:grid-cols-3">
        {LEAGUES.map((league) => (
          <Link
            key={league.id}
            to={league.href}
            aria-label={league.label}
            className="rounded-xl border border-white/10 px-6 py-8 text-center no-underline transition-colors hover:border-white/20 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
          >
            <span className="block text-lg font-semibold tracking-tight text-white">
              {league.label}
            </span>
            <span className="mt-2 block text-sm text-white/40">Matchups →</span>
          </Link>
        ))}
      </div>
    </section>
  );
}
