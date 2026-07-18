import type {
  ApiGameSlate,
  ApiHealth,
  ApiLeague,
  ApiLivePropsResponse,
  ApiLiveSlatesResponse,
  ApiPlayerProfile,
} from "@/types/api";

const API_BASE = "/api";

class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { Accept: "application/json" },
    ...init,
  });
  if (!res.ok) {
    throw new ApiError(res.status, `API ${path} failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export function todayIso(): string {
  const d = new Date();
  const y = d.getFullYear();
  const mo = String(d.getMonth() + 1).padStart(2, "0");
  const dy = String(d.getDate()).padStart(2, "0");
  return `${y}-${mo}-${dy}`;
}

export async function fetchHealth(): Promise<ApiHealth> {
  return apiFetch<ApiHealth>("/health");
}

export async function fetchGameSlate(date?: string): Promise<ApiGameSlate> {
  const d = date ?? todayIso();
  return apiFetch<ApiGameSlate>(`/games/${d}/slate`);
}

export async function fetchLiveProps(
  league: ApiLeague,
  date?: string,
): Promise<ApiLivePropsResponse> {
  const d = date ?? todayIso();
  const qs = new URLSearchParams({ league, date: d });
  return apiFetch<ApiLivePropsResponse>(`/live-props?${qs.toString()}`);
}

export async function fetchLiveSlates(
  league: ApiLeague,
  date?: string,
): Promise<ApiLiveSlatesResponse> {
  const d = date ?? todayIso();
  const qs = new URLSearchParams({ league, date: d });
  return apiFetch<ApiLiveSlatesResponse>(`/live-slates?${qs.toString()}`);
}

export async function fetchPlayerProfile(
  playerId: number,
  recentN = 10,
): Promise<ApiPlayerProfile> {
  return apiFetch<ApiPlayerProfile>(
    `/player/${playerId}?recent_n=${recentN}&include_predictions=true`,
  );
}

export async function isApiAvailable(): Promise<boolean> {
  try {
    const health = await fetchHealth();
    return health.status === "ok" || health.status === "healthy";
  } catch {
    return false;
  }
}

export { ApiError };
