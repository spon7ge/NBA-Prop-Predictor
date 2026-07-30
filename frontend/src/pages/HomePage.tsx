import { TicketHero } from "@/components/home/TicketHero";
import { LiveNowSection } from "@/components/home/LiveNowSection";
import { StoriesSection } from "@/components/home/StoriesSection";
import { ExploreSection } from "@/components/home/ExploreSection";
import { LearnTheGameSection } from "@/components/home/LearnTheGameSection";

export function HomePage() {
  return (
    <>
      <TicketHero />
      <LiveNowSection />
      <StoriesSection />
      <ExploreSection />
      <LearnTheGameSection />
    </>
  );
}
