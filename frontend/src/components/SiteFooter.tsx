const DISCLAIMER =
  "HoopVista provides research tools, projections, and statistics for informational and entertainment purposes only. Nothing on this site is gambling, financial, or legal advice. Sports betting involves risk of loss. You are solely responsible for your decisions and for complying with applicable laws in your jurisdiction. Past results do not guarantee future outcomes.";

const COPYRIGHT = "© Copyright 2026 HoopVista. All Rights Reserved.";

export function SiteFooter() {
  return (
    <footer className="mt-auto border-t border-white/10 bg-black px-4 py-8 sm:px-6">
      <div className="mx-auto max-w-5xl space-y-4 text-center">
        <p className="text-xs leading-relaxed text-white/45 sm:text-sm">
          {DISCLAIMER}
        </p>
        <p className="text-xs text-white/35">{COPYRIGHT}</p>
      </div>
    </footer>
  );
}
