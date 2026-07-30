import { ArrowUpRight } from "lucide-react";

type SourceBadge = {
  label: string;
  className: string;
};

type DataSource = {
  id: string;
  title: string;
  description: string;
  href: string;
  badges: SourceBadge[];
};

const SOURCES: DataSource[] = [
  {
    id: "nba-stats",
    title: "NBA Stats API",
    description:
      "Schedules, box scores, team and player game logs via nba_api.",
    href: "https://github.com/swar/nba_api",
    badges: [
      { label: "NBA", className: "bg-sky-600/25 text-sky-300" },
    ],
  },
  {
    id: "odds-api",
    title: "The Odds API",
    description: "Player prop lines for NBA and WNBA.",
    href: "https://the-odds-api.com",
    badges: [
      { label: "NBA", className: "bg-sky-600/25 text-sky-300" },
      { label: "WNBA", className: "bg-violet-600/25 text-violet-300" },
    ],
  },
  {
    id: "bref-wnba",
    title: "Basketball-Reference",
    description:
      "WNBA per-game and position tables used for player context.",
    href: "https://www.basketball-reference.com/wnba/",
    badges: [
      { label: "WNBA", className: "bg-violet-600/25 text-violet-300" },
    ],
  },
  {
    id: "supabase",
    title: "Supabase (PostgreSQL)",
    description: "Stored raw tables and engineered features.",
    href: "https://supabase.com",
    badges: [
      { label: "Shared", className: "bg-white/10 text-white/60" },
    ],
  },
];

export function DataSourcesSection() {
  return (
    <section className="mt-16 pb-8" aria-labelledby="about-data-sources">
      <h2
        id="about-data-sources"
        className="text-2xl font-semibold tracking-tight text-white sm:text-3xl"
      >
        Data sources
      </h2>
      <p className="mt-3 max-w-2xl text-sm leading-relaxed text-white/45">
        It&apos;s all public or licensed data feeds we use in the pipeline.
        Every chart should be able to say where its numbers came from.
      </p>

      <ul className="mt-8 divide-y divide-white/10 border-t border-white/10">
        {SOURCES.map((source) => (
          <li key={source.id}>
            <a
              href={source.href}
              target="_blank"
              rel="noopener noreferrer"
              className="group flex items-start justify-between gap-4 py-5 no-underline transition-colors hover:bg-white/[0.03]"
            >
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  {source.badges.map((badge) => (
                    <span
                      key={badge.label}
                      className={`rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-wide uppercase ${badge.className}`}
                    >
                      {badge.label}
                    </span>
                  ))}
                </div>
                <p className="mt-2 text-base font-semibold text-white">
                  {source.title}
                </p>
                <p className="mt-1 text-sm leading-relaxed text-white/45">
                  {source.description}
                </p>
              </div>
              <ArrowUpRight
                className="mt-1 size-4 shrink-0 text-white/35 transition-colors group-hover:text-white/70"
                aria-hidden
              />
            </a>
          </li>
        ))}
      </ul>
    </section>
  );
}
