import { TicketHero } from "@/components/home/TicketHero";
import { LiveNowSection } from "@/components/home/LiveNowSection";
import { StoriesSection } from "@/components/home/StoriesSection";
import { ExploreSection } from "@/components/home/ExploreSection";
import { LearnTheGameSection } from "@/components/home/LearnTheGameSection";
import { useWnbaScoreboard } from "@/hooks/useWnbaScoreboard";

export function HomePage() {
  const { liveGames, isLoading, hasNeverLoaded } = useWnbaScoreboard();
  return (
    <>
      <TicketHero />
      <LiveNowSection
        games={liveGames}
        isLoading={isLoading}
        isError={hasNeverLoaded}
      />
      <StoriesSection />
      <ExploreSection />
      <LearnTheGameSection />
    </>
  );
}
