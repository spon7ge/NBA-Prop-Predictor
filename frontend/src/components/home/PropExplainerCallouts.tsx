import { CALLOUTS, type CalloutId } from "./propExplainerDemo";

export type PropExplainerCalloutsProps = {
  layout: "desktop" | "mobile";
  /** Desktop only: which flank to render. Omit for mobile (all four). */
  slot?: "left" | "right";
};

const LEFT: CalloutId[] = ["line", "edge"];
const RIGHT: CalloutId[] = ["odds", "ev"];
const MOBILE: CalloutId[] = ["line", "odds", "edge", "ev"];

function idsFor(
  layout: "desktop" | "mobile",
  slot?: "left" | "right",
): CalloutId[] {
  if (layout === "mobile") return MOBILE;
  return slot === "right" ? RIGHT : LEFT;
}

export function PropExplainerCallouts({
  layout,
  slot,
}: PropExplainerCalloutsProps) {
  const ids = idsFor(layout, slot);

  return (
    <div
      className={
        layout === "mobile" ? "flex flex-col gap-4" : "flex flex-col gap-6"
      }
    >
      {ids.map((id) => {
        const { title, body } = CALLOUTS[id];

        return (
          <div
            key={id}
            data-testid={`callout-${id}`}
            className={
              layout === "mobile"
                ? "border-l-2 border-emerald-300/50 pl-3 opacity-100"
                : "border-b border-dotted border-emerald-300/50 pb-1 opacity-100"
            }
          >
            <h4 className="text-sm font-medium text-white">{title}</h4>
            <p className="mt-1 text-sm text-white/50">{body}</p>
          </div>
        );
      })}
    </div>
  );
}
