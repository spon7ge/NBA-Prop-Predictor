type SectionHeadingProps = {
  title: string;
  /** Optional muted subtitle shown between the title and the divider. */
  subtitle?: string;
};

/**
 * Shared home-section header: uppercase title, optional subtitle, then a
 * hairline rule that fills the remaining row (matches Learn the Game).
 */
export function SectionHeading({ title, subtitle }: SectionHeadingProps) {
  return (
    <div className="mb-5 flex items-center gap-4">
      <h2 className="shrink-0 text-sm font-bold tracking-[0.15em] text-white/55 uppercase">
        {title}
      </h2>
      {subtitle ? (
        <p className="shrink-0 text-sm text-white/40">{subtitle}</p>
      ) : null}
      <div className="h-px flex-1 bg-white/10" aria-hidden />
    </div>
  );
}
