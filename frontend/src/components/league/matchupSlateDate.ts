const ET = "America/New_York";
export const SLATE_ROLLOVER_HOUR_ET = 3;
const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

export function slateEtDate(now: Date = new Date()): string {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: ET,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "numeric",
    hour12: false,
  }).formatToParts(now);
  const get = (type: string) =>
    Number(parts.find((p) => p.type === type)?.value);
  const year = get("year");
  const month = get("month");
  const day = get("day");
  let hour = get("hour");
  // en-US hourCycle can yield 24 for midnight — normalize
  if (hour === 24) hour = 0;
  const iso = `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
  if (hour < SLATE_ROLLOVER_HOUR_ET) {
    return shiftEtDate(iso, -1);
  }
  return iso;
}

export function shiftEtDate(dateIso: string, deltaDays: number): string {
  const [y, m, d] = dateIso.split("-").map(Number);
  const utc = new Date(Date.UTC(y!, (m ?? 1) - 1, d ?? 1));
  utc.setUTCDate(utc.getUTCDate() + deltaDays);
  return utc.toISOString().slice(0, 10);
}

export function isValidEtDate(
  value: string | null | undefined,
): value is string {
  if (!value || !DATE_RE.test(value)) return false;
  const [y, m, d] = value.split("-").map(Number);
  const dt = new Date(Date.UTC(y!, (m ?? 1) - 1, d ?? 1));
  return dt.toISOString().slice(0, 10) === value;
}

export function parseMatchupDateParam(
  raw: string | null,
  today: string,
): string {
  return isValidEtDate(raw) ? raw : today;
}

export function formatMatchupNavLabel(
  dateIso: string,
  today: string,
): string {
  if (dateIso === today) return "Today";
  const date = new Date(`${dateIso}T12:00:00-04:00`);
  return new Intl.DateTimeFormat("en-US", {
    timeZone: ET,
    month: "short",
    day: "numeric",
  }).format(date);
}

export function isOddsWindowDate(
  selectedDate: string,
  today: string,
): boolean {
  return (
    selectedDate === today ||
    selectedDate === shiftEtDate(today, 1) ||
    selectedDate === shiftEtDate(today, 2)
  );
}
