import type { ApiGradedParlay } from "@/types/api";
import { dfsNetPayoutMult } from "@/lib/constants";
import { formatUsd } from "@/lib/legsRoi";

interface TicketStripProps {
  parlays: ApiGradedParlay[];
  stakePerTicket: number;
  bookLabel: (raw: string) => string;
}

function legTone(hit: boolean | null): "hit" | "miss" | "pending" {
  if (hit === true) return "hit";
  if (hit === false) return "miss";
  return "pending";
}

export function TicketStripList({
  parlays,
  stakePerTicket,
  bookLabel,
}: TicketStripProps) {
  const decided = parlays.filter((p) => p.cashed != null);
  if (decided.length === 0) {
    return (
      <p className="results-viz-empty">No settled tickets to strip yet.</p>
    );
  }

  return (
    <ul className="results-ticket-strips" aria-label="Ticket outcomes by leg">
      {decided.slice(0, 20).map((p, i) => {
        const delta = p.cashed
          ? stakePerTicket * dfsNetPayoutMult(p.bookmaker, p.n_legs)
          : -stakePerTicket;
        const kill = p.legs.find((l) => l.hit === false);
        return (
          <li
            key={`${p.game_date}-${p.bookmaker}-${p.n_legs}-${i}`}
            className={`results-ticket-strip${
              p.cashed ? " results-ticket-strip--cash" : " results-ticket-strip--miss"
            }`}
          >
            <div className="results-ticket-strip-meta">
              <span className="results-ticket-strip-status">
                {p.cashed ? "CASH" : "MISS"}
              </span>
              <span className="results-ticket-strip-book">
                {bookLabel(p.bookmaker)} · {p.n_legs}-leg
              </span>
              <span
                className={`results-ticket-strip-delta${
                  delta >= 0
                    ? " results-roi-stat-value--pos"
                    : " results-roi-stat-value--neg"
                }`}
              >
                {formatUsd(delta)}
              </span>
            </div>
            <div
              className="results-ticket-strip-bar"
              role="img"
              aria-label={`${p.n_legs} legs: ${p.legs_hit} hit`}
            >
              {p.legs.map((leg, j) => {
                const tone = legTone(leg.hit);
                return (
                  <span
                    key={`${leg.player_name}-${j}`}
                    className={`results-ticket-seg results-ticket-seg--${tone}`}
                    title={`${leg.player_name} ${leg.side} ${leg.market} ${leg.line ?? ""} → ${
                      leg.actual_stat ?? leg.miss_reason ?? "—"
                    }`}
                  />
                );
              })}
            </div>
            {!p.cashed && kill && (
              <p className="results-ticket-strip-kill">
                Killed by {kill.player_name}
                {kill.team_abbr ? ` (${kill.team_abbr})` : ""} · {kill.side}{" "}
                {kill.market} {kill.line}
                {kill.actual_stat != null ? ` → ${kill.actual_stat}` : ""}
              </p>
            )}
          </li>
        );
      })}
    </ul>
  );
}
