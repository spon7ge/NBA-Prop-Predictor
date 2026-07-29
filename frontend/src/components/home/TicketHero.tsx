import { Link } from "react-router-dom";
import {
  ArrowDown,
  CircleDot,
  Dribbble,
  Ticket,
  Trophy,
} from "lucide-react";

const ringIcons = [
  { Icon: Dribbble, color: "text-orange-400", angle: 0 },
  { Icon: Trophy, color: "text-amber-300", angle: 36 },
  { Icon: CircleDot, color: "text-sky-400", angle: 72 },
  { Icon: Dribbble, color: "text-rose-400", angle: 108 },
  { Icon: Trophy, color: "text-emerald-400", angle: 144 },
  { Icon: CircleDot, color: "text-violet-400", angle: 180 },
  { Icon: Dribbble, color: "text-cyan-400", angle: 216 },
  { Icon: Trophy, color: "text-yellow-400", angle: 252 },
  { Icon: CircleDot, color: "text-pink-400", angle: 288 },
  { Icon: Dribbble, color: "text-lime-400", angle: 324 },
] as const;

export function TicketHero() {
  return (
    <section className="mx-auto max-w-6xl px-4 py-8 sm:px-6 sm:py-10">
      <div
        className="relative overflow-hidden rounded-2xl border border-white/10 bg-[#141414] shadow-[0_0_0_1px_rgba(255,255,255,0.04),0_0_40px_rgba(255,255,255,0.04)]"
        style={{
          backgroundImage:
            "radial-gradient(rgba(255,255,255,0.07) 1px, transparent 1px)",
          backgroundSize: "18px 18px",
        }}
      >
        <div className="grid gap-10 px-6 py-10 sm:px-10 sm:py-12 lg:grid-cols-[1.1fr_0.9fr] lg:items-center lg:gap-6">
          <div className="space-y-5">
            <p className="text-[11px] font-medium tracking-[0.2em] text-white/45 uppercase">
              Season Pass
            </p>
            <div className="flex flex-wrap items-center gap-3">
              <h1 className="text-4xl font-bold tracking-tight text-white sm:text-5xl">
                hoopvista
              </h1>
              <span className="rounded-full border border-white/20 px-2.5 py-0.5 text-[10px] font-semibold tracking-wider text-white/70 uppercase">
                Beta
              </span>
            </div>
            <p className="text-lg text-white/85 sm:text-xl">
              your seat to every game.
            </p>
            <p className="max-w-md text-sm leading-relaxed text-white/50">
              Props, edges, and clear explanations — so you know what you&apos;re
              looking at before tip-off.
            </p>
            <div className="flex flex-wrap gap-3 pt-1">
              <a
                href="#live-now"
                className="inline-flex items-center gap-2 rounded-full bg-white px-5 py-2.5 text-sm font-semibold text-black no-underline transition-opacity hover:opacity-90"
              >
                See what&apos;s inside
                <ArrowDown className="size-4" aria-hidden />
              </a>
              <Link
                to="/slate"
                className="inline-flex items-center rounded-full border border-white/25 bg-transparent px-5 py-2.5 text-sm font-semibold text-white no-underline transition-colors hover:bg-white/5"
              >
                Learn the games
              </Link>
            </div>
          </div>

          <div className="relative mx-auto flex size-56 items-center justify-center sm:size-64">
            {ringIcons.map(({ Icon, color, angle }) => {
              const rad = (angle * Math.PI) / 180;
              const radius = 96;
              const x = Math.cos(rad) * radius;
              const y = Math.sin(rad) * radius;
              return (
                <span
                  key={`${angle}-${color}`}
                  className={`absolute flex size-8 items-center justify-center rounded-full border border-white/10 bg-[#1a1a1a] ${color}`}
                  style={{
                    transform: `translate(${x}px, ${y}px)`,
                  }}
                  aria-hidden
                >
                  <Icon className="size-3.5" />
                </span>
              );
            })}
            <div className="relative z-10 flex size-16 items-center justify-center rounded-xl border border-white/15 bg-[#0f0f0f] text-white shadow-lg">
              <Ticket className="size-7" aria-hidden />
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between border-t border-dashed border-white/15 px-6 py-3 font-mono text-[10px] tracking-wider text-white/40 uppercase sm:px-10">
          <span>Gate 6 · Sec 102 · Row 6 · Seat 1</span>
          <span>Admit One</span>
        </div>
      </div>
    </section>
  );
}
