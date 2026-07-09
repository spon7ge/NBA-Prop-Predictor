"""Deprecated — use scripts/run_transforms.py instead."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    print(
        "run_dbt.py is deprecated. Use: python scripts/run_transforms.py",
        file=sys.stderr,
    )
    args = argv if argv is not None else sys.argv[1:]
    select = None
    run_test = False
    i = 0
    while i < len(args):
        if args[i] == "run":
            i += 1
            continue
        if args[i] == "test":
            run_test = True
            i += 1
            continue
        if args[i] == "--select" and i + 1 < len(args):
            select = args[i + 1]
            i += 2
            continue
        i += 1

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.utils.transforms import assert_ml_predictions_model_id, run_transforms

    if run_test:
        assert_ml_predictions_model_id()
        return 0
    run_transforms(select=select)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
