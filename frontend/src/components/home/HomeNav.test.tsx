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

  it("points league links at /#live-now", () => {
    renderNav("/about");
    expect(screen.getByRole("link", { name: "NBA" })).toHaveAttribute(
      "href",
      "/#live-now",
    );
    expect(screen.getByRole("link", { name: "WNBA" })).toHaveAttribute(
      "href",
      "/#live-now",
    );
  });
});
