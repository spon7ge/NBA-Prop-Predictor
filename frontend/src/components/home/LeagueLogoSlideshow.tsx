import { useEffect, useState } from "react";
import basketball from "@/assets/basketball.png";
import wnbaBasketball from "@/assets/wnba_basketball.png";

const SLIDES = [
  { src: basketball, alt: "NBA" },
  { src: wnbaBasketball, alt: "WNBA" },
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
        className="relative flex min-h-52 items-center justify-center sm:min-h-64"
        aria-label="League logos"
      >
        <img
          src={slide.src}
          alt={slide.alt}
          className="size-28 object-contain sm:size-32"
        />
      </div>
    );
  }

  return (
    <div
      className="relative flex min-h-52 items-center justify-center sm:min-h-64"
      aria-label="League logos"
    >
      {SLIDES.map((slide, index) => (
        <img
          key={slide.alt}
          src={slide.src}
          alt={slide.alt}
          className={
            index === 0
              ? "league-logo-slide league-logo-slide-a absolute size-28 object-contain sm:size-32"
              : "league-logo-slide league-logo-slide-b absolute size-28 object-contain sm:size-32"
          }
        />
      ))}
    </div>
  );
}
