const STACK_ROWS = [
  {
    id: "frontend",
    label: "FRONTEND",
    items:
      "React 19, TypeScript, Vite, Tailwind CSS v4, React Router, Geist, lucide-react, Vitest",
  },
  {
    id: "backend",
    label: "BACKEND",
    items:
      "Python, FastAPI, Pandas, NumPy, scikit-learn, XGBoost, nba_api, SQLAlchemy, Supabase/PostgreSQL, joblib",
  },
  {
    id: "infra",
    label: "INFRA & TOOLING",
    items: "GitHub Actions, GitHub Pages",
  },
] as const;

export function TechStackSection() {
  return (
    <section className="mt-16" aria-labelledby="about-tech-stack">
      <h2
        id="about-tech-stack"
        className="text-2xl font-semibold tracking-tight text-white sm:text-3xl"
      >
        Tech stack
      </h2>

      <div className="mt-6 overflow-hidden rounded-2xl border border-white/10 bg-[#141414]">
        <dl className="divide-y divide-white/10">
          {STACK_ROWS.map((row) => (
            <div
              key={row.id}
              className="grid gap-2 px-5 py-4 sm:grid-cols-[10rem_1fr] sm:gap-6 sm:px-6"
            >
              <dt className="font-mono text-[11px] font-medium tracking-wide text-white/40">
                {row.label}
              </dt>
              <dd className="font-mono text-sm leading-relaxed text-white/70">
                {row.items}
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}
