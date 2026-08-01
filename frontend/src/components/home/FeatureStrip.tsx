const FEATURES = [
  {
    title: "Reference",
    body: "Live games, standings, leaders, and box scores — the full basketball picture.",
  },
  {
    title: "Lines",
    body: "Player props and market odds in one quiet board, not ten open tabs.",
  },
  {
    title: "Projections",
    body: "Model numbers next to the books — so your bets start with an edge.",
  },
] as const;

export function FeatureStrip() {
  return (
    <section
      id="built-for-clarity"
      className="mx-auto max-w-6xl border-t border-white/10 px-4 py-16 sm:px-6 sm:py-20"
    >
      <div className="mx-auto max-w-3xl text-center">
        <h2 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">
          The stats site. The betting edge. Together.
        </h2>
        <p className="mt-3 text-sm text-white/40 sm:text-base">
          Follow the game like a reference site. Bet like you have a model.
        </p>
      </div>
      <div className="mx-auto mt-12 grid max-w-4xl gap-10 text-left sm:grid-cols-3 sm:gap-8">
        {FEATURES.map((feature) => (
          <div key={feature.title}>
            <h3 className="text-sm font-semibold text-white">{feature.title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-white/40">
              {feature.body}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}
