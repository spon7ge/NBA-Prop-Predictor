import { useState } from "react";
import { Link } from "react-router-dom";
import { PropExplainerCard } from "./PropExplainerCard";
import { PropExplainerCallouts } from "./PropExplainerCallouts";
import type { PropSide } from "./propExplainerDemo";

export function PropExplainerSection() {
  const [selectedSide, setSelectedSide] = useState<PropSide>("over");

  return (
    <section
      id="how-a-prop-works"
      className="mx-auto max-w-6xl border-t border-white/10 px-4 py-16 sm:px-6 sm:py-20"
    >
      <div className="mx-auto max-w-3xl text-center">
        <p className="text-[11px] font-medium tracking-[0.18em] text-white/40 uppercase">
          How a prop works
        </p>
        <h2 className="mt-3 text-2xl font-semibold tracking-tight text-white sm:text-3xl">
          Read the line. See the edge.
        </h2>
        <p className="mt-3 text-sm text-white/40 sm:text-base">
          Model projection next to the market — tap Over or Under to see how EV
          flips.
        </p>
      </div>

      <div className="mx-auto mt-12 hidden max-w-5xl items-center gap-4 lg:grid lg:grid-cols-[1fr_minmax(260px,320px)_1fr]">
        <PropExplainerCallouts
          selectedSide={selectedSide}
          layout="desktop"
          slot="left"
        />
        <PropExplainerCard
          selectedSide={selectedSide}
          onSelectSide={setSelectedSide}
        />
        <PropExplainerCallouts
          selectedSide={selectedSide}
          layout="desktop"
          slot="right"
        />
      </div>

      <div className="mx-auto mt-10 max-w-md lg:hidden">
        <PropExplainerCard
          selectedSide={selectedSide}
          onSelectSide={setSelectedSide}
        />
        <div className="mt-4">
          <PropExplainerCallouts
            selectedSide={selectedSide}
            layout="mobile"
          />
        </div>
      </div>

      <div className="mx-auto mt-12 max-w-2xl text-center">
        <p className="text-sm leading-relaxed text-white/40 sm:text-base">
          We project the player. Books set the line. Positive EV means the model
          likes your side more than the price implies.
        </p>
        <div className="mt-6">
          <Link
            to="/wnba/prop_picks"
            className="inline-flex items-center gap-2 rounded-full bg-white px-5 py-2.5 text-sm font-semibold text-black no-underline transition-opacity hover:opacity-90 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
          >
            See live props
          </Link>
        </div>
      </div>
    </section>
  );
}
