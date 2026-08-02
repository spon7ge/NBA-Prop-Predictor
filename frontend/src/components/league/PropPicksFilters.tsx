import { useEffect, useId, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import { TeamAbbrevAvatar } from "@/components/TeamAbbrevAvatar";
import {
  PROP_BOOK_OPTIONS,
  type TeamFilterOption,
} from "./filterPropLines";

type MultiSelectFilterProps = {
  label: string;
  options: { value: string; label: string; logoUrl?: string | null }[];
  selected: Set<string>;
  onChange: (next: Set<string>) => void;
};

function MultiSelectFilter({
  label,
  options,
  selected,
  onChange,
}: MultiSelectFilterProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const listId = useId();
  const triggerLabel =
    selected.size > 0 ? `${label} (${selected.size})` : label;

  useEffect(() => {
    if (!open) return;
    function onPointerDown(event: MouseEvent) {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  function toggle(value: string) {
    const next = new Set(selected);
    if (next.has(value)) next.delete(value);
    else next.add(value);
    onChange(next);
  }

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={listId}
        onClick={() => setOpen((v) => !v)}
        className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium transition-colors ${
          selected.size > 0
            ? "border-white/20 bg-white/10 text-white"
            : "border-white/10 bg-transparent text-white/55 hover:text-white"
        }`}
      >
        {triggerLabel}
        <ChevronDown className="size-3 opacity-70" aria-hidden strokeWidth={1.75} />
      </button>
      {open ? (
        <ul
          id={listId}
          role="listbox"
          aria-label={label}
          aria-multiselectable="true"
          className="absolute top-full left-0 z-20 mt-1.5 max-h-56 min-w-[10rem] overflow-y-auto rounded-lg border border-white/10 bg-black py-0.5"
        >
          {options.length === 0 ? (
            <li className="px-2.5 py-1.5 text-[11px] text-white/40">No options</li>
          ) : (
            options.map((opt) => {
              const checked = selected.has(opt.value);
              return (
                <li key={opt.value}>
                  <button
                    type="button"
                    role="option"
                    aria-selected={checked}
                    className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-xs text-white/80 hover:bg-white/5"
                    onClick={() => toggle(opt.value)}
                  >
                    <span
                      className={`flex size-4 shrink-0 items-center justify-center rounded border text-[10px] ${
                        checked
                          ? "border-white/40 bg-white/15 text-white"
                          : "border-white/20 bg-transparent text-transparent"
                      }`}
                      aria-hidden
                    >
                      ✓
                    </span>
                    {opt.logoUrl !== undefined ? (
                      <TeamAbbrevAvatar
                        abbrev={opt.value}
                        logoUrl={opt.logoUrl}
                        sizeClassName="size-5"
                      />
                    ) : null}
                    <span>{opt.label}</span>
                  </button>
                </li>
              );
            })
          )}
        </ul>
      ) : null}
    </div>
  );
}

export type PropPicksFiltersProps = {
  stats: string[];
  teams: TeamFilterOption[];
  selectedStats: Set<string>;
  selectedSides: Set<string>;
  selectedTeams: Set<string>;
  selectedBooks: Set<string>;
  onStatsChange: (next: Set<string>) => void;
  onSidesChange: (next: Set<string>) => void;
  onTeamsChange: (next: Set<string>) => void;
  onBooksChange: (next: Set<string>) => void;
  onClear: () => void;
};

export function PropPicksFilters({
  stats,
  teams,
  selectedStats,
  selectedSides,
  selectedTeams,
  selectedBooks,
  onStatsChange,
  onSidesChange,
  onTeamsChange,
  onBooksChange,
  onClear,
}: PropPicksFiltersProps) {
  const hasActive =
    selectedStats.size > 0 ||
    selectedSides.size > 0 ||
    selectedTeams.size > 0 ||
    selectedBooks.size > 0;

  return (
    <div
      className="flex flex-wrap items-center gap-2"
      aria-label="Prop picks filters"
    >
      <MultiSelectFilter
        label="Book"
        options={PROP_BOOK_OPTIONS.map((b) => ({
          value: b.key,
          label: b.label,
        }))}
        selected={selectedBooks}
        onChange={onBooksChange}
      />
      <MultiSelectFilter
        label="Stat"
        options={stats.map((s) => ({ value: s, label: s }))}
        selected={selectedStats}
        onChange={onStatsChange}
      />
      <MultiSelectFilter
        label="O/U"
        options={[
          { value: "over", label: "Over" },
          { value: "under", label: "Under" },
        ]}
        selected={selectedSides}
        onChange={onSidesChange}
      />
      <MultiSelectFilter
        label="Team"
        options={teams.map((t) => ({
          value: t.abbrev,
          label: t.abbrev,
          logoUrl: t.logoUrl,
        }))}
        selected={selectedTeams}
        onChange={onTeamsChange}
      />
      <button
        type="button"
        disabled
        className="cursor-not-allowed rounded-md border border-white/10 px-2.5 py-1 text-xs font-medium text-white/25"
        title="Coming soon"
      >
        +EV · Soon
      </button>
      {hasActive ? (
        <button
          type="button"
          onClick={onClear}
          className="px-1.5 text-xs text-white/40 transition-colors hover:text-white"
        >
          Clear filters
        </button>
      ) : null}
    </div>
  );
}
