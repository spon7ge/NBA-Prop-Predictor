import { useEffect, useRef, useState } from "react";

interface DropdownOption<T extends string | number> {
  value: T;
  label: string;
}

interface DropdownProps<T extends string | number> {
  id: string;
  label: string;
  value: T;
  options: DropdownOption<T>[];
  onChange: (value: T) => void;
  classPrefix: "book" | "legs";
}

export function Dropdown<T extends string | number>({
  id,
  label,
  value,
  options,
  onChange,
  classPrefix,
}: DropdownProps<T>) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const selected = options.find((o) => o.value === value);

  useEffect(() => {
    function onDocClick() {
      setOpen(false);
    }
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("click", onDocClick);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("click", onDocClick);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, []);

  const triggerId = `${id}Trigger`;
  const menuId = `${id}Menu`;

  return (
    <div className={`${classPrefix}-filter`}>
      <label className={`${classPrefix}-filter-label`} id={`${id}Label`} htmlFor={triggerId}>
        {label}
      </label>
      <div className={`${classPrefix}-dropdown`} id={id} ref={rootRef} onClick={(e) => e.stopPropagation()}>
        <button
          type="button"
          className={`${classPrefix}-dropdown-trigger${open ? ` ${classPrefix}-dropdown-trigger--open` : ""}`}
          id={triggerId}
          aria-haspopup="listbox"
          aria-expanded={open}
          aria-controls={menuId}
          onClick={() => setOpen((v) => !v)}
        >
          <span className={`${classPrefix}-dropdown-value`}>{selected?.label ?? ""}</span>
          <span className={`${classPrefix}-dropdown-chevron`} aria-hidden="true" />
        </button>
        <div
          className={`${classPrefix}-dropdown-menu`}
          id={menuId}
          role="listbox"
          hidden={!open}
          aria-labelledby={`${id}Label`}
        >
          {options.map((opt) => {
            const current = opt.value === value;
            return (
              <button
                key={String(opt.value)}
                type="button"
                className={`${classPrefix}-dropdown-option${current ? ` ${classPrefix}-dropdown-option--current` : ""}`}
                role="option"
                aria-selected={current}
                onClick={() => {
                  onChange(opt.value);
                  setOpen(false);
                }}
              >
                {opt.label}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
