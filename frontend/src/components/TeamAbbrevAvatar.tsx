import { useState } from "react";

type TeamAbbrevAvatarProps = {
  abbrev: string;
  logoUrl: string | null;
  /** Tailwind size classes for the circle/img, e.g. `size-7` or `size-8`. */
  sizeClassName?: string;
};

export function TeamAbbrevAvatar({
  abbrev,
  logoUrl,
  sizeClassName = "size-7",
}: TeamAbbrevAvatarProps) {
  const [failed, setFailed] = useState(false);
  const letter = abbrev.slice(0, 1);
  const showLogo = Boolean(logoUrl) && !failed;

  if (showLogo && logoUrl) {
    return (
      <img
        src={logoUrl}
        alt=""
        role="presentation"
        className={`${sizeClassName} shrink-0 object-contain`}
        onError={() => setFailed(true)}
      />
    );
  }

  return (
    <span
      className={`flex ${sizeClassName} shrink-0 items-center justify-center rounded-full bg-white/10 text-[10px] font-bold text-white/70`}
    >
      {letter}
    </span>
  );
}
