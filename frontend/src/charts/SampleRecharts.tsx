import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

const DATA = [
  { name: "Mon", pts: 12 },
  { name: "Tue", pts: 18 },
  { name: "Wed", pts: 9 },
  { name: "Thu", pts: 22 },
  { name: "Fri", pts: 15 },
];

export function SampleRecharts() {
  if (!DATA.length) {
    return <p className="text-sm text-muted-foreground">No chart data.</p>;
  }

  return (
    <div className="h-56 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={DATA}>
          <CartesianGrid stroke="currentColor" className="text-border" vertical={false} />
          <XAxis dataKey="name" tick={{ fill: "currentColor" }} className="text-muted-foreground text-xs" />
          <YAxis tick={{ fill: "currentColor" }} className="text-muted-foreground text-xs" />
          <Tooltip />
          <Bar dataKey="pts" fill="oklch(0.45 0 0)" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
