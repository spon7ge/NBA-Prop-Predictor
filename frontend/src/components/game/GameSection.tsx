import type { ComponentProps } from "react";

/** Quiet surface matching homepage Live now cards. */
export const GAME_SECTION_SURFACE =
  "rounded-xl border border-white/10 bg-white/[0.03] p-4";

export function GameSection({
  children,
  className = "",
  ...props
}: ComponentProps<"section">) {
  return (
    <section
      className={`${GAME_SECTION_SURFACE} ${className}`.trim()}
      {...props}
    >
      {children}
    </section>
  );
}
