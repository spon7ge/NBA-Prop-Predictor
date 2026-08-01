import { ArrowDown } from "lucide-react";
import { LeagueLogoSlideshow } from "./LeagueLogoSlideshow";

export function BrandHero() {
  return (
    <section className="mx-auto max-w-6xl px-4 py-16 sm:px-6 sm:py-20 lg:py-24">
      <div className="grid items-center gap-12 lg:grid-cols-2 lg:gap-16">
        <div className="space-y-5 text-left">
          <p className="text-[11px] font-medium tracking-[0.18em] text-white/40 uppercase">
            Stats + edges · Beta
          </p>
          <h1 className="text-4xl font-semibold tracking-tight text-white sm:text-5xl">
            HoopVista
          </h1>
          <p className="text-xl tracking-tight text-white/85 sm:text-2xl">
            Basketball Reference meets BettingPros — in one place.
          </p>
          <p className="max-w-md text-sm leading-relaxed text-white/45 sm:text-base">
            Box scores, standings, and leaders for fans who want the full
            picture — plus model projections next to the lines when you&apos;re
            ready to bet smarter.
          </p>
          <div className="pt-1">
            <a
              href="#live-now"
              className="inline-flex items-center gap-2 rounded-full bg-white px-5 py-2.5 text-sm font-semibold text-black no-underline transition-opacity hover:opacity-90 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
            >
              See what&apos;s live
              <ArrowDown className="size-4" aria-hidden strokeWidth={1.75} />
            </a>
          </div>
        </div>

        <LeagueLogoSlideshow />
      </div>
    </section>
  );
}
