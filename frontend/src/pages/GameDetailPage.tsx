import { Link, useParams } from "react-router-dom";
import { useGameDetail } from "@/hooks/useGameDetail";
import { mapGameDetail } from "@/components/game/mapGameDetail";
import { GameHeader } from "@/components/game/GameHeader";
import { ShotChart } from "@/components/game/ShotChart";
import { PlayByPlay } from "@/components/game/PlayByPlay";

function GameDetailSkeleton() {
  return (
    <div className="mx-auto max-w-6xl space-y-4 px-4 py-6 sm:px-6" aria-hidden>
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="h-4 w-16 animate-pulse rounded bg-white/10" />
          <span className="h-3 w-20 animate-pulse rounded bg-white/10" />
        </div>
        <div className="rounded-xl border border-white/10 bg-[#141414] p-4">
          <span className="mb-3 block h-3 w-40 animate-pulse rounded bg-white/10" />
          <div className="space-y-3">
            {[0, 1].map((row) => (
              <div key={row} className="flex items-center justify-between gap-3">
                <span className="h-4 w-40 animate-pulse rounded bg-white/10" />
                <span className="size-12 shrink-0 animate-pulse rounded-md bg-white/10" />
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <span className="h-80 animate-pulse rounded-xl border border-white/10 bg-[#141414]" />
        <span className="h-80 animate-pulse rounded-xl border border-white/10 bg-[#141414]" />
      </div>
    </div>
  );
}

export function GameDetailPage() {
  const { espnEventId } = useParams<{ espnEventId: string }>();
  const { data, isLoading, hasNeverLoaded } = useGameDetail(espnEventId);

  if (hasNeverLoaded) {
    return (
      <div className="mx-auto max-w-6xl space-y-3 px-4 py-10 text-center sm:px-6">
        <p className="text-sm text-white/60">Unable to load game</p>
        <Link
          to="/"
          className="inline-flex items-center gap-1 text-sm font-medium text-white/70 no-underline transition-colors hover:text-white"
        >
          ← Back
        </Link>
      </div>
    );
  }

  if (isLoading && !data) {
    return <GameDetailSkeleton />;
  }

  if (!data) {
    return null;
  }

  const detail = mapGameDetail(data);

  return (
    <div className="mx-auto max-w-6xl space-y-4 px-4 py-6 sm:px-6">
      <GameHeader detail={detail} />
      <div className="grid gap-4 lg:grid-cols-2">
        <ShotChart detail={detail} />
        <PlayByPlay detail={detail} />
      </div>
    </div>
  );
}
