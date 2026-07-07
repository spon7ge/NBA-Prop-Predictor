import type { ReactNode } from "react";
interface SkeletonBlockProps {
  className?: string;
  width?: string;
  height?: string;
}

function SkeletonBlock({ className = "", width, height = "14px" }: SkeletonBlockProps) {
  return (
    <span
      className={`skeleton-block ${className}`.trim()}
      style={{ width, height }}
      aria-hidden="true"
    />
  );
}

export function LoadingMessage({ children }: { children: ReactNode }) {
  return (
    <div className="loading-panel" role="status" aria-live="polite">
      <div className="loading-spinner" aria-hidden="true" />
      <p className="load-msg">{children}</p>
    </div>
  );
}

export function PlayerBlockSkeleton() {
  return (
    <div className="player-block player-block--skeleton" aria-hidden="true">
      <div className="player-row player-row--skeleton">
        <div className="player-row-main">
          <SkeletonBlock width="140px" height="16px" />
          <SkeletonBlock width="90px" height="12px" className="skeleton-mt" />
        </div>
        <div className="player-row-stats">
          <SkeletonBlock width="100px" height="12px" />
          <SkeletonBlock width="60px" height="18px" />
        </div>
      </div>
    </div>
  );
}

export function PlayersListSkeleton({ count = 6 }: { count?: number }) {
  return (
    <div className="players-grouped">
      <div className="players-grouped-header">
        <SkeletonBlock width="120px" height="14px" />
        <SkeletonBlock width="200px" height="28px" />
      </div>
      {Array.from({ length: count }, (_, i) => (
        <PlayerBlockSkeleton key={i} />
      ))}
    </div>
  );
}

export function PlayerStatsSkeleton() {
  return (
    <div className="player-stats-panel player-stats-panel--skeleton" aria-hidden="true">
      <SkeletonBlock width="160px" height="14px" className="skeleton-mb" />
      <div className="player-stats-grid">
        {Array.from({ length: 4 }, (_, i) => (
          <SkeletonBlock key={i} width="100%" height="32px" />
        ))}
      </div>
    </div>
  );
}

export function ErrorPanel({
  message,
  onRetry,
}: {
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div className="error-panel" role="alert">
      <p className="load-msg load-err">{message}</p>
      {onRetry && (
        <button type="button" className="retry-btn" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}
