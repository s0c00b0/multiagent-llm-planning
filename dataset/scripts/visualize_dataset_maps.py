"""Generate map visualizations for each entry in dataset/maps using coin_collector.visualize_map."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MAPS_DIR = DATASET_DIR / "maps"
VIZ_DIR = DATASET_DIR / "visualization"
sys.path.insert(0, str(PROJECT_ROOT))

from src.coin_collector import CoinCollectorGame, visualize_map


def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    index_path = MAPS_DIR / "index.json"
    if not index_path.exists():
        print("Run dataset/scripts/build_dataset.py first.")
        sys.exit(1)
    with open(index_path) as f:
        maps = json.load(f)
    for entry in maps:
        num_coins = entry.get("num_coins", 2 if entry["num_locations"] == 10 else (1 if entry["num_locations"] <= 5 else 3))
        game = CoinCollectorGame(
            num_locations=entry["num_locations"],
            num_distractor_items=entry["num_distractor_items"],
            include_doors=entry["include_doors"],
            num_coins=num_coins,
            connectivity=entry["connectivity"],
            coins_in_containers=entry["coins_in_containers"],
            seed=entry["seed"],
            topology=entry.get("topology"),
        )
        game.reset(seed=entry["seed"])
        out = VIZ_DIR / f"{entry['id']}.png"
        visualize_map(game, output_path=str(out), show_player_locations=True, show_coin_locations=True)
        print(f"  {out.relative_to(PROJECT_ROOT)}")
    print(f"Done. {len(maps)} maps in {VIZ_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
