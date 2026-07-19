import type { View } from "@/types/slate";

interface HeaderProps {
  activeView: View;
  onViewChange: (view: View) => void;
}

export function Header({ activeView, onViewChange }: HeaderProps) {
  return (
    <header className="slate-header">
      <div className="slate-header-row">
        <h1 className="slate-title">
          <a href="./" className="slate-title-link">
            HoopVista
          </a>
        </h1>
        <nav className="header-links slate-header-links" aria-label="Site navigation">
          <a href="blog.html" className="header-link">
            Blog
          </a>
          <a href="contact.html" className="header-link">
            Contact
          </a>
          <a href="faq.html" className="header-link">
            FAQ
          </a>
          <button
            type="button"
            className="header-link"
            aria-current={activeView === "results" ? "page" : undefined}
            onClick={() => onViewChange("results")}
          >
            Results
          </button>
          <button
            type="button"
            className="header-link"
            aria-current={activeView === "pairs" ? "page" : undefined}
            onClick={() => onViewChange("pairs")}
          >
            Top Legs
          </button>
          <button
            type="button"
            className="header-link"
            aria-current={activeView === "players" ? "page" : undefined}
            onClick={() => onViewChange("players")}
          >
            All Players
          </button>
        </nav>
      </div>
    </header>
  );
}
