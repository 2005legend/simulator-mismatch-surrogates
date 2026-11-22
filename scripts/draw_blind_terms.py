"""Addendum v1.10, part A3: draw the missing terms from the hash of the commit tagged prereg-v1.10a.

Usage:  python -m scripts.draw_blind_terms        (writes pilot_results/blind_draw.json)
"""

import json
import os
import subprocess

from mismatch.blind import draw

TAG = "prereg-v1.10a"


def main():
    h = subprocess.run(["git", "rev-list", "-n", "1", TAG], capture_output=True, text=True, check=True).stdout.strip()
    out = {"tag": TAG, "commit": h, **draw(h)}
    os.makedirs("pilot_results", exist_ok=True)
    with open(os.path.join("pilot_results", "blind_draw.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
