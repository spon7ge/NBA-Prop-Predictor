import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";

export function NotFoundPage() {
  return (
    <main className="mx-auto flex min-h-svh max-w-lg flex-col justify-center gap-4 px-6">
      <h1 className="text-2xl font-bold tracking-tight">Page not found</h1>
      <p className="text-muted-foreground">That route does not exist.</p>
      <Link
        to="/"
        className="inline-flex items-center gap-1 text-sm font-medium text-foreground hover:underline"
      >
        Back to home
        <ArrowRight className="size-4" />
      </Link>
    </main>
  );
}
