import type { ApiWnbaGameDetail } from "@/lib/api";
import type { GameDetail } from "./types";

export function mapGameDetail(detail: ApiWnbaGameDetail): GameDetail {
  return {
    espnEventId: detail.espn_event_id,
    league: detail.league,
    status: detail.status,
    statusLabel: detail.status_label,
    venue: detail.venue,
    away: {
      id: detail.away.id,
      abbrev: detail.away.abbrev,
      name: detail.away.name,
      score: detail.away.score,
      color: detail.away.color,
    },
    home: {
      id: detail.home.id,
      abbrev: detail.home.abbrev,
      name: detail.home.name,
      score: detail.home.score,
      color: detail.home.color,
    },
    fgMade: detail.fg_made,
    fgAttempted: detail.fg_attempted,
    latestPlay: detail.latest_play
      ? {
          id: detail.latest_play.id,
          clock: detail.latest_play.clock,
          period: detail.latest_play.period,
          text: detail.latest_play.text,
          teamId: detail.latest_play.team_id,
        }
      : null,
    shots: detail.shots.map((shot) => ({
      id: shot.id,
      teamId: shot.team_id,
      playerName: shot.player_name,
      made: shot.made,
      x: shot.x,
      y: shot.y,
      period: shot.period,
      clock: shot.clock,
    })),
    plays: detail.plays.map((play) => ({
      id: play.id,
      teamId: play.team_id,
      period: play.period,
      clock: play.clock,
      text: play.text,
      scoring: play.scoring,
      awayScore: play.away_score,
      homeScore: play.home_score,
      shooting: play.shooting,
    })),
    winProbability: detail.win_probability
      ? {
          summary: detail.win_probability.summary,
          timeline: detail.win_probability.timeline.map((point) => ({
            id: point.id,
            period: point.period,
            clock: point.clock,
            awayScore: point.away_score,
            homeScore: point.home_score,
            awayWinPct: point.away_win_pct,
            homeWinPct: point.home_win_pct,
            teamId: point.team_id,
          })),
          teamStats: detail.win_probability.team_stats.map((stat) => ({
            key: stat.key,
            label: stat.label,
            awayValue: stat.away_value,
            homeValue: stat.home_value,
          })),
        }
      : null,
  };
}
