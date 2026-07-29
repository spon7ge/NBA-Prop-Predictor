import { useQuery } from "@tanstack/react-query";

export const labDemoQueryKey = ["lab", "demo"] as const;

export type LabDemoDatum = { label: string; value: number };

export async function fetchLabDemo(): Promise<LabDemoDatum> {
  // Isolated demo — no slate/props APIs
  await new Promise((r) => setTimeout(r, 200));
  return { label: "stack-ready", value: 1 };
}

export function useLabDemoQuery() {
  return useQuery({
    queryKey: labDemoQueryKey,
    queryFn: fetchLabDemo,
  });
}
