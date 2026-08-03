import { useEffect, useState } from "react";
import nbaLogo from "@/assets/nba_logo.png";
import wnbaLogo from "@/assets/wnba_logo.png";

const MLB_LOGO =
  "https://a.espncdn.com/i/teamlogos/leagues/500/mlb.png";

const SLIDES = [
  { src: nbaLogo, alt: "NBA" },
  { src: wnbaLogo, alt: "WNBA" },
  { src: MLB_LOGO, alt: "MLB" },
] as const;

const SLIDE_ANIMATION = [
  "league-logo-slide-a",
  "league-logo-slide-b",
  "league-logo-slide-c",
] as const;

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(() =>
    typeof window !== "undefined"
      ? window.matchMedia("(prefers-reduced-motion: reduce)").matches
      : false,
  );

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = () => setReduced(mq.matches);
    onChange();
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  return reduced;
}

export function LeagueLogoSlideshow() {
  const reducedMotion = usePrefersReducedMotion();

  if (reducedMotion) {
    const slide = SLIDES[0];
    return (
      <div
        className="relative flex min-h-72 items-center justify-center sm:min-h-80"
        aria-label="League logos"
      >
        <img
          src={slide.src}
          alt={slide.alt}
          className="size-48 object-contain sm:size-56 lg:size-64"
        />
      </div>
    );
  }

  return (
    <div
      className="relative flex min-h-72 items-center justify-center sm:min-h-80"
      aria-label="League logos"
    >
      {SLIDES.map((slide, index) => (
        <img
          key={slide.alt}
          src={slide.src}
          alt={slide.alt}
          className={`league-logo-slide ${SLIDE_ANIMATION[index]} absolute size-48 object-contain sm:size-56 lg:size-64`}
        />
      ))}
    </div>
  );
}
