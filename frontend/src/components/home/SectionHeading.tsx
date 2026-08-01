type SectionHeadingProps = {
  title: string;
  /** Optional muted subtitle shown next to the title. */
  subtitle?: string;
};

/**
 * Shared home-section header: display title, optional muted subtitle.
 */
export function SectionHeading({ title, subtitle }: SectionHeadingProps) {
  return (
    <div className="mb-8 flex flex-wrap items-baseline gap-x-3 gap-y-1">
      <h2 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">
        {title}
      </h2>
      {subtitle ? (
        <p className="text-sm text-white/40">{subtitle}</p>
      ) : null}
    </div>
  );
}
