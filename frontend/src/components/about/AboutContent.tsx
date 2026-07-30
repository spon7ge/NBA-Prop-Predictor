const LEAGUES = [
  {
    id: "nba",
    label: "NBA",
    className: "border-sky-500/60 text-sky-300",
  },
  {
    id: "wnba",
    label: "WNBA",
    className: "border-violet-500/60 text-violet-300",
  },
] as const;

export function AboutContent() {
  return (
    <main className="mx-auto max-w-3xl px-4 py-16 sm:px-6">
      <p className="inline-flex rounded-full border border-white/20 px-3 py-1 text-[11px] font-medium tracking-wide text-white/50 uppercase">
        Sports Analytics · Beta
      </p>

      <h1 className="mt-6 font-heading text-4xl font-semibold tracking-tight text-white sm:text-5xl">
        About HoopVista.
      </h1>

      <ul className="mt-6 flex flex-wrap gap-2" aria-label="Leagues">
        {LEAGUES.map((league) => (
          <li
            key={league.id}
            className={`rounded-full border px-3 py-1 text-xs font-medium ${league.className}`}
          >
            {league.label}
          </li>
        ))}
      </ul>

      <div className="mt-8 space-y-5 text-base leading-relaxed text-white/55">
        <p>
          HoopVista is an interactive basketball analytics site for the NBA and
          WNBA — live trackers, props context, and visualizations that help you
          see the game from a better seat.
        </p>
        <p>
          We design for any fan. Every stat comes with a plain-language
          explainer, and charts stay honest about what the numbers do and do not
          say.
        </p>
        <p>
          The site is still in beta. We are actively adding tools and polishing
          what is already here.
        </p>
      </div>
    </main>
  );
}
