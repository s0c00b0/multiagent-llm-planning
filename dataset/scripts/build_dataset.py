"""Build dataset maps (20 JSONs + index) from dataset_selected.json or defaults, then regenerate visualizations."""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MAPS_DIR = DATASET_DIR / "maps"
SELECTED_PATH = Path(__file__).resolve().parent / "dataset_selected.json"

DEFAULT_MAPS = [
    {"id": "5room_01", "num_locations": 5, "seed": 101, "connectivity": 0.15, "topology": "linear"},
    {"id": "5room_02", "num_locations": 5, "seed": 102, "connectivity": 0.08, "topology": "straight line"},
    {"id": "5room_03", "num_locations": 5, "seed": 103, "connectivity": 0.45, "topology": "one cycle"},
    {"id": "5room_04", "num_locations": 5, "seed": 104, "connectivity": 0.5, "topology": "branching"},
    {"id": "5room_05", "num_locations": 5, "seed": 105, "connectivity": 0.7, "topology": "hub and spokes"},
    {"id": "10room_01", "num_locations": 10, "seed": 201, "connectivity": 0.15, "topology": "linear"},
    {"id": "10room_02", "num_locations": 10, "seed": 202, "connectivity": 0.1, "topology": "straight line"},
    {"id": "10room_03", "num_locations": 10, "seed": 203, "connectivity": 0.4, "topology": "one cycle"},
    {"id": "10room_04", "num_locations": 10, "seed": 204, "connectivity": 0.55, "topology": "two cycles"},
    {"id": "10room_05", "num_locations": 10, "seed": 205, "connectivity": 0.35, "topology": "branchy path"},
    {"id": "10room_06", "num_locations": 10, "seed": 206, "connectivity": 0.6, "topology": "complex branching"},
    {"id": "10room_07", "num_locations": 10, "seed": 207, "connectivity": 0.7, "topology": "grid-like"},
    {"id": "10room_08", "num_locations": 10, "seed": 208, "connectivity": 0.5, "topology": "tree with cycles"},
    {"id": "10room_09", "num_locations": 10, "seed": 209, "connectivity": 0.45, "topology": "chain with side branches"},
    {"id": "10room_10", "num_locations": 10, "seed": 210, "connectivity": 0.85, "topology": "dense mesh"},
    {"id": "15room_01", "num_locations": 15, "seed": 301, "connectivity": 0.12, "topology": "linear"},
    {"id": "15room_02", "num_locations": 15, "seed": 302, "connectivity": 0.4, "topology": "long cycle"},
    {"id": "15room_03", "num_locations": 15, "seed": 303, "connectivity": 0.55, "topology": "multi-branch"},
    {"id": "15room_04", "num_locations": 15, "seed": 304, "connectivity": 0.75, "topology": "complex mesh"},
    {"id": "15room_05", "num_locations": 15, "seed": 305, "connectivity": 0.9, "topology": "highly connected"},
]


def load_maps():
    if SELECTED_PATH.exists():
        with open(SELECTED_PATH) as f:
            return json.load(f)
    return DEFAULT_MAPS


def _num_coins_for_size(num_locations: int) -> int:
    if num_locations <= 5:
        return 1
    if num_locations <= 10:
        return 2
    return 3


def build_entry(m: dict) -> dict:
    n = m["num_locations"]
    return {
        "id": m["id"],
        "num_locations": n,
        "num_coins": _num_coins_for_size(n),
        "seed": m["seed"],
        "connectivity": m["connectivity"],
        "include_doors": True,
        "num_distractor_items": 2,
        "coins_in_containers": False,
        "topology": m["topology"],
    }


def main():
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    maps = load_maps()
    index = []
    for m in maps:
        entry = build_entry(m)
        path = MAPS_DIR / f"{m['id']}.json"
        with open(path, "w") as f:
            json.dump(entry, f, indent=2)
        index.append(entry)
    index_path = MAPS_DIR / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote {len(maps)} maps to {MAPS_DIR}")

    viz_script = Path(__file__).resolve().parent / "visualize_dataset_maps.py"
    if viz_script.exists():
        print("Regenerating visualizations...")
        r = subprocess.run(
            [sys.executable, str(viz_script)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            if r.stdout:
                print(r.stdout.strip())
        else:
            print("Visualization failed (e.g. matplotlib not installed):", r.stderr or r.stdout or r.returncode, file=sys.stderr)


if __name__ == "__main__":
    main()
