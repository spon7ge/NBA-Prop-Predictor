import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { HomeNav } from "./HomeNav";

function renderNav(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <HomeNav />
    </MemoryRouter>,
  );
}

describe("HomeNav", () => {
  it("labels the primary nav and hides only league links on mobile", () => {
    renderNav("/");

    expect(
      screen.getByRole("navigation", { name: "Primary" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "NBA" }).parentElement).toHaveClass(
      "hidden",
      "sm:flex",
    );
    expect(screen.getByRole("link", { name: "About" }).parentElement).not.toHaveClass(
      "hidden",
    );
  });

  it("links About to /about", () => {
    renderNav("/");
    expect(screen.getByRole("link", { name: "About" })).toHaveAttribute(
      "href",
      "/about",
    );
  });

  it("marks About as current on /about", () => {
    renderNav("/about");
    expect(screen.getByRole("link", { name: "About" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  it("does not mark About current on home", () => {
    renderNav("/");
    expect(screen.getByRole("link", { name: "About" })).not.toHaveAttribute(
      "aria-current",
    );
  });

  it("points league links at matchups hubs", () => {
    renderNav("/about");
    expect(screen.getByRole("link", { name: "NBA" })).toHaveAttribute(
      "href",
      "/nba/matchups",
    );
    expect(screen.getByRole("link", { name: "WNBA" })).toHaveAttribute(
      "href",
      "/wnba/matchups",
    );
  });

  it("marks WNBA current on /wnba/matchups", () => {
    renderNav("/wnba/matchups");
    expect(screen.getByRole("link", { name: "WNBA" })).toHaveAttribute(
      "aria-current",
      "page",
    );
    expect(screen.getByRole("link", { name: "NBA" })).not.toHaveAttribute(
      "aria-current",
    );
  });

  it("marks NBA current on /nba/matchups", () => {
    renderNav("/nba/matchups");
    expect(screen.getByRole("link", { name: "NBA" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  it("uses official league logos in the nav", () => {
    const { container } = renderNav("/");
    const images = container.querySelectorAll('nav img[aria-hidden="true"]');
    expect(images).toHaveLength(2);
    expect(images[0]?.getAttribute("src")).toMatch(/nba_logo/);
    expect(images[1]?.getAttribute("src")).toMatch(/wnba_logo/);
  });
});
