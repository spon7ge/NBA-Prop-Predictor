const DATA_PREFIX = import.meta.env.DEV ? "../data" : "./data";

function dataPath(...parts: string[]): string {
  return [DATA_PREFIX, ...parts].join("/");
}

async function fetchWithFallback<T>(
  urls: string[],
  parse: (res: Response) => Promise<T>,
  fallback: T,
): Promise<T> {
  for (const url of urls) {
    try {
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await parse(res);
    } catch {
      /* try next */
    }
  }
  return fallback;
}

export function jsonUrls(filename: string): string[] {
  const primary = dataPath("props/ev_analysis", filename);
  const secondary = import.meta.env.DEV
    ? `./data/props/ev_analysis/${filename}`
    : `../data/props/ev_analysis/${filename}`;
  return [primary, secondary];
}

export function enrichedUrls(): string[] {
  const urls: string[] = [];
  const latest = "dfs_enriched_latest.json";
  urls.push(dataPath("props/enriched", latest));
  urls.push(
    import.meta.env.DEV
      ? `./data/props/enriched/${latest}`
      : `../data/props/enriched/${latest}`,
  );

  const now = new Date();
  for (let i = 0; i < 7; i += 1) {
    const d = new Date(now.getFullYear(), now.getMonth(), now.getDate() - i);
    const y = d.getFullYear();
    const mo = String(d.getMonth() + 1).padStart(2, "0");
    const dy = String(d.getDate()).padStart(2, "0");
    const fname = `dfs_enriched_${y}${mo}${dy}.json`;
    urls.push(dataPath("props/enriched", fname));
    urls.push(
      import.meta.env.DEV
        ? `./data/props/enriched/${fname}`
        : `../data/props/enriched/${fname}`,
    );
  }
  return urls;
}

export async function fetchSlateJson<T>(urls: string[]): Promise<T[]> {
  return fetchWithFallback(
    urls,
    async (res) => {
      const data = (await res.json()) as unknown;
      return Array.isArray(data) ? (data as T[]) : [];
    },
    [],
  );
}

export async function fetchEnrichedPicks(): Promise<unknown[]> {
  return fetchWithFallback(
    enrichedUrls(),
    async (res) => {
      const data = (await res.json()) as { picks?: unknown[] };
      return Array.isArray(data.picks) ? data.picks : [];
    },
    [],
  );
}
