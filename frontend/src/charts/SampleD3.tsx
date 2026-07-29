import { useEffect, useRef } from "react";
import * as d3 from "d3";

const VALUES = [4, 8, 15, 16, 23, 42];

export function SampleD3() {
  const ref = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    if (!VALUES.length) return;

    const width = 320;
    const height = 120;
    const margin = { top: 8, right: 8, bottom: 24, left: 28 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3
      .scaleBand()
      .domain(VALUES.map((_, i) => String(i)))
      .range([0, innerW])
      .padding(0.2);

    const y = d3
      .scaleLinear()
      .domain([0, d3.max(VALUES) ?? 0])
      .nice()
      .range([innerH, 0]);

    g.selectAll("rect")
      .data(VALUES)
      .join("rect")
      .attr("x", (_, i) => x(String(i)) ?? 0)
      .attr("y", (d) => y(d))
      .attr("width", x.bandwidth())
      .attr("height", (d) => innerH - y(d))
      .attr("fill", "oklch(0.45 0 0)")
      .attr("rx", 3);

    g.append("g")
      .attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(x).tickFormat((d) => String(Number(d) + 1)))
      .attr("color", "oklch(0.55 0 0)");

    g.append("g")
      .call(d3.axisLeft(y).ticks(4))
      .attr("color", "oklch(0.55 0 0)");
  }, []);

  if (!VALUES.length) {
    return <p className="text-sm text-muted-foreground">No chart data.</p>;
  }

  return <svg ref={ref} className="w-full max-w-sm" role="img" aria-label="D3 sample bar chart" />;
}
