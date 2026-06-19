"""Export NYK_Analysis.ipynb chart outputs to docs/images/nyk/ for the blog.

Usage (from repo root):
  1. Run all cells in notebooks/NYK_Analysis.ipynb
  2. python scripts/export_nyk_blog_charts.py
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / 'notebooks' / 'NYK_Analysis.ipynb'
OUT = ROOT / 'docs' / 'images' / 'nyk'

# notebook cell index -> PNG filenames (one per plt.show() / display in that cell)
CELL_EXPORTS = {
    7: ['quarter-splits'],
    10: ['clutch-time'],
    13: ['efficiency-wl', 'efficiency-net-rating'],
    16: ['comeback-frequency'],
}


def main() -> None:
    nb = json.loads(NB.read_text(encoding='utf-8'))
    OUT.mkdir(parents=True, exist_ok=True)

    for cell_idx, filenames in CELL_EXPORTS.items():
        cell = nb['cells'][cell_idx]
        png_outputs = [
            o for o in cell.get('outputs', [])
            if o.get('output_type') == 'display_data' and 'image/png' in o.get('data', {})
        ]
        if len(png_outputs) != len(filenames):
            raise RuntimeError(
                f'Cell {cell_idx}: expected {len(filenames)} chart(s), found {len(png_outputs)}. '
                'Re-run the notebook first.'
            )
        for name, output in zip(filenames, png_outputs):
            path = OUT / f'{name}.png'
            path.write_bytes(base64.b64decode(output['data']['image/png']))
            print(f'Wrote {path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
