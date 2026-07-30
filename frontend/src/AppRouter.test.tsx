import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { AppRouter } from "@/AppRouter";

describe("AppRouter", () => {
  it("renders home at /", () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <AppRouter />
      </MemoryRouter>,
    );
    expect(
      screen.getByRole("heading", { name: /hoopvista/i }),
    ).toBeInTheDocument();
  });

  it("renders about at /about", () => {
    render(
      <MemoryRouter initialEntries={["/about"]}>
        <AppRouter />
      </MemoryRouter>,
    );
    expect(
      screen.getByRole("heading", { name: /about hoopvista/i }),
    ).toBeInTheDocument();
    expect(screen.getByText("No live games")).toBeInTheDocument();
  });

  it("renders not found for unknown paths", () => {
    render(
      <MemoryRouter initialEntries={["/slate"]}>
        <AppRouter />
      </MemoryRouter>,
    );
    expect(
      screen.getByRole("heading", { name: /page not found/i }),
    ).toBeInTheDocument();
  });
});
