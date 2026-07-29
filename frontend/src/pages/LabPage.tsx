import { Link } from "react-router-dom";
import { ArrowRight, ArrowUpRight, FlaskConical } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { SampleRecharts } from "@/charts/SampleRecharts";
import { SampleD3 } from "@/charts/SampleD3";
import { useLabDemoQuery } from "@/lib/labDemoQuery";

export function LabPage() {
  const demo = useLabDemoQuery();

  return (
    <main className="mx-auto max-w-3xl space-y-10 px-6 py-12">
      <header className="space-y-3">
        <div className="flex items-center gap-2">
          <FlaskConical className="size-5 text-muted-foreground" />
          <span className="rounded-md bg-league-badge px-2 py-0.5 text-xs font-medium text-league-badge-foreground">
            NBA
          </span>
        </div>
        <h1 className="text-3xl font-bold tracking-tight">Stack lab</h1>
        <p className="text-muted-foreground">
          Tailwind v4, shadcn, Query, Recharts, and D3 — ready for product UI.
        </p>
        <Link
          to="/"
          className="inline-flex items-center gap-1 text-sm font-medium hover:underline"
        >
          Open dashboard
          <ArrowRight className="size-4" />
        </Link>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Controls</CardTitle>
          <CardDescription>shadcn Button + arrow CTA pattern</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-3">
          <Button type="button">Primary</Button>
          <Button type="button" variant="outline">
            Outline
          </Button>
          <Button type="button" variant="ghost" className="gap-1">
            External pattern
            <ArrowUpRight className="size-4" />
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>TanStack Query</CardTitle>
          <CardDescription>Isolated demo key — not slate/props</CardDescription>
        </CardHeader>
        <CardContent>
          {demo.isPending && (
            <p className="text-sm text-muted-foreground">Loading…</p>
          )}
          {demo.isError && (
            <p className="text-sm text-muted-foreground">Demo query failed.</p>
          )}
          {demo.isSuccess && (
            <p className="text-sm">
              <span className="font-medium">{demo.data.label}</span>
              <span className="text-muted-foreground"> · value {demo.data.value}</span>
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Recharts</CardTitle>
          <CardDescription>Muted neutral series</CardDescription>
        </CardHeader>
        <CardContent>
          <SampleRecharts />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>D3</CardTitle>
          <CardDescription>Small SVG sample</CardDescription>
        </CardHeader>
        <CardContent>
          <SampleD3 />
        </CardContent>
      </Card>
    </main>
  );
}
