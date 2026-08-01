import {
  CALLOUTS,
  isCalloutEmphasized,
  type CalloutId,
  type PropSide,
} from "./propExplainerDemo";

export type PropExplainerCalloutsProps = {
  selectedSide: PropSide;
  layout: "desktop" | "mobile";
  /** Desktop only: which flank to render. Omit for mobile (all four). */
  slot?: "left" | "right";
};

const LEFT: CalloutId[] = ["line", "edge"];
const RIGHT: CalloutId[] = ["side", "flip"];
const MOBILE: CalloutId[] = ["line", "side", "edge", "flip"];

function idsFor(
  layout: "desktop" | "mobile",
  slot?: "left" | "right",
): CalloutId[] {
  if (layout === "mobile") return MOBILE;
  return slot === "right" ? RIGHT : LEFT;
}

function calloutClassName(
  emphasized: boolean,
  layout: "desktop" | "mobile",
  selectedSide: PropSide,
): string {
  const opacity = emphasized ? "opacity-100" : "opacity-35";

  if (!emphasized) return opacity;

  if (layout === "mobile") {
    const accent =
      selectedSide === "over"
        ? "border-l-2 border-emerald-300/50"
        : "border-l-2 border-red-300/50";
    return `${opacity} ${accent} pl-3`;
  }

  const leader =
    selectedSide === "over"
      ? "border-b border-dotted border-emerald-300/50"
      : "border-b border-dotted border-red-300/50";
  return `${opacity} ${leader} pb-1`;
}

export function PropExplainerCallouts({
  selectedSide,
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
        const emphasized = isCalloutEmphasized(id, selectedSide);
        const { title, body } = CALLOUTS[id];

        return (
          <div
            key={id}
            data-testid={`callout-${id}`}
            data-emphasized={String(emphasized)}
            className={calloutClassName(emphasized, layout, selectedSide)}
          >
            <h4 className="text-sm font-medium text-white">{title}</h4>
            <p className="mt-1 text-sm text-white/50">{body}</p>
          </div>
        );
      })}
    </div>
  );
}
