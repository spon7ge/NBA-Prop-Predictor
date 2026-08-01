const FEATURES = [
  {
    title: "Follow",
    body: "Live scores and tonight’s slate — a calm place to watch the game.",
  },
  {
    title: "Understand",
    body: "Props and matchups explained simply, whether you’re new or deep in.",
  },
  {
    title: "Decide",
    body: "Model projections next to the market — so your bets start with an edge.",
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
          Built to get you ready
        </h2>
        <p className="mt-3 text-sm text-white/40 sm:text-base">
          Watch first. Learn the lines. Bet with a plan.
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
