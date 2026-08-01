import { BrandHero } from "@/components/home/BrandHero";
import { LiveNowSection } from "@/components/home/LiveNowSection";
import { StoriesSection } from "@/components/home/StoriesSection";
import { FeatureStrip } from "@/components/home/FeatureStrip";
import { PropExplainerSection } from "@/components/home/PropExplainerSection";
import { LeagueCtaSection } from "@/components/home/LeagueCtaSection";
import { useWnbaScoreboard } from "@/hooks/useWnbaScoreboard";

export function HomePage() {
  const { liveGames, isLoading, hasNeverLoaded } = useWnbaScoreboard();
  return (
    <>
      <BrandHero />
      <LiveNowSection
        games={liveGames}
        isLoading={isLoading}
        isError={hasNeverLoaded}
      />
      <StoriesSection />
      <FeatureStrip />
      <PropExplainerSection />
      <LeagueCtaSection />
    </>
  );
}
