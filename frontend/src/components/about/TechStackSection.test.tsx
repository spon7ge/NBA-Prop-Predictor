import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { TechStackSection } from "./TechStackSection";

describe("TechStackSection", () => {
  it("renders three stack rows with current-truth tech", () => {
    render(<TechStackSection />);

    expect(
      screen.getByRole("heading", { name: "Tech stack" }),
    ).toBeInTheDocument();
    expect(screen.getByText("FRONTEND")).toBeInTheDocument();
    expect(screen.getByText("BACKEND")).toBeInTheDocument();
    expect(screen.getByText("INFRA & TOOLING")).toBeInTheDocument();
    expect(screen.getByText(/React 19/)).toBeInTheDocument();
    expect(screen.getByText(/FastAPI/)).toBeInTheDocument();
    expect(screen.getByText(/GitHub Pages/)).toBeInTheDocument();
    expect(screen.queryByText(/system design/i)).not.toBeInTheDocument();
  });
});
