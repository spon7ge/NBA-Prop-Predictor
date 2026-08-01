const FEATURES = [
  {
    title: "Props",
    body: "Player lines from the books that matter, in one place.",
  },
  {
    title: "Edges",
    body: "Where the model and the market disagree.",
  },
  {
    title: "Explain",
    body: "Plain-language why — not just a number.",
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
          Built for clarity
        </h2>
        <p className="mt-3 text-sm text-white/40 sm:text-base">
          Three things we surface before every tip-off.
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
