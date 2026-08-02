import { GraduationCap } from "lucide-react";
import type { HomeLeague, LearnSport } from "./types";
import { normalizeLiveGames } from "./format";
import { SectionHeading } from "./SectionHeading";

type LearnTheGameSectionProps = {
  sports?: LearnSport[];
};

const leagueStyles: Record<
  HomeLeague,
  { badge: string; link: string }
> = {
  nba: {
    badge: "bg-sky-600/25 text-sky-300",
    link: "text-sky-400 hover:text-sky-300",
  },
  wnba: {
    badge: "bg-violet-600/25 text-violet-300",
    link: "text-violet-400 hover:text-violet-300",
  },
  mlb: {
    badge: "bg-emerald-600/25 text-emerald-300",
    link: "text-emerald-400 hover:text-emerald-300",
  },
};

export const DEFAULT_LEARN_SPORTS: LearnSport[] = [
  { id: "nba-basketball", league: "nba", sport: "Basketball", href: "#" },
  { id: "wnba-basketball", league: "wnba", sport: "Basketball", href: "#" },
];

function resolveSports(sports: LearnSport[] | undefined): LearnSport[] {
  if (sports === undefined) return DEFAULT_LEARN_SPORTS;
  return normalizeLiveGames(sports);
}

function SportCard({ sport }: { sport: LearnSport }) {
  const styles = leagueStyles[sport.league];

  return (
    <article className="rounded-xl border border-white/10 bg-[#1a1a1a] p-4">
      <span
        className={`inline-block rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-wide uppercase ${styles.badge}`}
      >
        {sport.league.toUpperCase()}
      </span>
      <p className="mt-3 text-sm font-semibold text-white">{sport.sport}</p>
      <a
        href={sport.href ?? "#"}
        className={`mt-2 inline-flex text-sm font-medium no-underline transition-colors ${styles.link}`}
      >
        How it works →
      </a>
    </article>
  );
}

export function LearnTheGameSection({ sports }: LearnTheGameSectionProps) {
  const list = resolveSports(sports);

  return (
    <section id="learn-the-game" className="mx-auto max-w-6xl px-4 py-10 pb-16 sm:px-6">
      <SectionHeading title="Learn the Game" />

      <div className="rounded-2xl border border-white/10 bg-[#141414] p-6 sm:p-8">
        <div className="mb-6 flex size-9 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-white/80">
          <GraduationCap className="size-4" aria-hidden />
        </div>
        <h3 className="text-xl font-semibold tracking-tight text-white sm:text-2xl">
          New to a sport? Start here.
        </h3>
        <p className="mt-3 max-w-2xl text-sm leading-relaxed text-white/50">
          Plain-English walkthroughs of how each game works, with interactive
          bits you can play with. And any stat with a dotted underline explains
          itself when you tap it.
        </p>

        {list.length === 0 ? (
          <p className="mt-8 text-sm text-white/40">No primers yet.</p>
        ) : (
          <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {list.map((sport) => (
              <SportCard key={sport.id} sport={sport} />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
