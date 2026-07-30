import { Outlet } from "react-router-dom";
import { HomeNav } from "@/components/home/HomeNav";
import { LiveTicker } from "@/components/home/LiveTicker";

export function HomeChromeLayout() {
  return (
    <div className="min-h-screen bg-black text-white">
      <HomeNav />
      <LiveTicker />
      <main>
        <Outlet />
      </main>
    </div>
  );
}
