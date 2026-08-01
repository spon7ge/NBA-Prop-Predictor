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
        className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-sm font-medium ${
          selected.size > 0
            ? "border-violet-500/50 bg-violet-600/20 text-violet-100"
            : "border-white/10 bg-white/[0.03] text-white/70 hover:text-white"
        }`}
      >
        {triggerLabel}
        <ChevronDown className="size-3.5 opacity-70" aria-hidden />
      </button>
      {open ? (
        <ul
          id={listId}
          role="listbox"
          aria-label={label}
          aria-multiselectable="true"
          className="absolute top-full left-0 z-20 mt-2 max-h-64 min-w-[11rem] overflow-y-auto rounded-xl border border-white/10 bg-[#121212] py-1 shadow-xl"
        >
          {options.length === 0 ? (
            <li className="px-3 py-2 text-xs text-white/40">No options</li>
          ) : (
            options.map((opt) => {
              const checked = selected.has(opt.value);
              return (
                <li key={opt.value}>
                  <button
                    type="button"
                    role="option"
                    aria-selected={checked}
                    className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-white/80 hover:bg-white/5"
                    onClick={() => toggle(opt.value)}
                  >
                    <span
                      className={`flex size-4 shrink-0 items-center justify-center rounded border text-[10px] ${
                        checked
                          ? "border-violet-400 bg-violet-600 text-white"
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
        className="cursor-not-allowed rounded-full border border-white/10 bg-white/[0.03] px-3 py-1.5 text-sm font-medium text-white/35"
        title="Coming soon"
      >
        +EV · Soon
      </button>
      {hasActive ? (
        <button
          type="button"
          onClick={onClear}
          className="px-2 text-sm text-white/50 hover:text-white"
        >
          Clear filters
        </button>
      ) : null}
    </div>
  );
}
