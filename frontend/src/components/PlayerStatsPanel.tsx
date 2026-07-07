import { fmt1, fmtNumOrDash } from "@/lib/format";
import { usePlayerProfile } from "@/lib/queries";
import { PlayerStatsSkeleton } from "@/components/LoadingSkeleton";

const PROP_LABELS: Record<string, string> = {
  min: "Min",
  ppm: "PPM",
  rpm: "RPM",
  apm: "APM",
};

interface PlayerStatsPanelProps {
  playerId: number | undefined;
  enabled: boolean;
}

export function PlayerStatsPanel({ playerId, enabled }: PlayerStatsPanelProps) {
  const { data, isLoading, isError } = usePlayerProfile(playerId, enabled);

  if (!enabled || !playerId) return null;

  if (isLoading) return <PlayerStatsSkeleton />;

  if (isError || !data) {
    return (
      <div className="player-stats-panel player-stats-panel--error">
        <p className="load-msg">Could not load player stats.</p>
      </div>
    );
  }

  const l5 = data.rolling_avg_5;
  const l10 = data.rolling_avg_10;
  const recent = data.recent_games.slice(0, 5);

  return (
    <div className="player-stats-panel">
      <h3 className="player-stats-title">Player Stats</h3>

      {(l5 || l10) && (
        <div className="player-stats-rolling">
          {l5 && (
            <div className="player-stats-chip">
              <span className="player-stats-chip-label">L5 avg</span>
              <span className="player-stats-chip-value">
                {fmt1(l5.pts_roll5)} PTS · {fmt1(l5.reb_roll5)} REB · {fmt1(l5.ast_roll5)} AST
              </span>
            </div>
          )}
          {l10 && (
            <div className="player-stats-chip">
              <span className="player-stats-chip-label">L10 avg</span>
              <span className="player-stats-chip-value">
                {fmt1(l10.pts_roll10)} PTS · {fmt1(l10.reb_roll10)} REB · {fmt1(l10.ast_roll10)} AST
              </span>
            </div>
          )}
        </div>
      )}

      {data.predictions.length > 0 && (
        <div className="player-stats-predictions">
          <span className="player-stats-subtitle">Model predictions</span>
          <div className="player-stats-pred-grid">
            {data.predictions.map((p) => (
              <div key={`${p.prop}-${p.game_id}`} className="player-stats-pred">
                <span className="player-stats-pred-label">{PROP_LABELS[p.prop] ?? p.prop}</span>
                <span className="player-stats-pred-value">{fmtNumOrDash(p.prediction)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {recent.length > 0 && (
        <div className="player-stats-recent">
          <span className="player-stats-subtitle">Recent games</span>
          <div className="players-wrap">
            <table className="players-table player-stats-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Opp</th>
                  <th className="num">MIN</th>
                  <th className="num">PTS</th>
                  <th className="num">REB</th>
                  <th className="num">AST</th>
                  <th className="num">+/-</th>
                </tr>
              </thead>
              <tbody>
                {recent.map((g) => (
                  <tr key={g.game_id}>
                    <td>{g.game_date}</td>
                    <td>{g.opp_team_abbreviation ?? g.matchup ?? "—"}</td>
                    <td className="num">{fmt1(g.min)}</td>
                    <td className="num">{fmt1(g.pts)}</td>
                    <td className="num">{fmt1(g.reb)}</td>
                    <td className="num">{fmt1(g.ast)}</td>
                    <td className="num">{fmtNumOrDash(g.plus_minus)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
