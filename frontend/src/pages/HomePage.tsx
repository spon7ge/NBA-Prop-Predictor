import { HomeNav } from "@/components/home/HomeNav";
import { LiveTicker } from "@/components/home/LiveTicker";
import { TicketHero } from "@/components/home/TicketHero";
import { LiveNowSection } from "@/components/home/LiveNowSection";

export function HomePage() {
  return (
    <div className="min-h-screen bg-black text-white">
      <HomeNav />
      <LiveTicker />
      <TicketHero />
      <LiveNowSection />
    </div>
  );
}
