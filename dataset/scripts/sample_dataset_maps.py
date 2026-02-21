"""
Sample up to 100000 seeds per map size to find maps where coins are far from start
so most of the map is utilized. Outputs dataset_selected.json for build_dataset.py.
"""

import json
import random
import sys
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.coin_collector.game import CoinCollectorGame


def get_room_for_coin(coin):
    c = coin.current_container
    if c is None:
        return None
    if getattr(c, "is_location", False):
        return c
    if c.current_container and getattr(c.current_container, "is_location", False):
        return c.current_container
    return c.current_container


def distances_from_start(locations):
    if not locations:
        return {}
    start = locations[0]
    dist = {start: 0}
    q = deque([start])
    while q:
        r = q.popleft()
        d = dist[r]
        for neighbor in (r.location_north, r.location_south, r.location_east, r.location_west):
            if neighbor is not None and neighbor not in dist:
                dist[neighbor] = d + 1
                q.append(neighbor)
    return dist


def evaluate_map(game):
    game.reset(seed=game.seed)
    locations = game.locations
    task_objects = game.task_objects
    if not locations or not task_objects:
        return None
    dist_map = distances_from_start(locations)
    n = len(locations)
    coin_distances = []
    for coin in task_objects:
        room = get_room_for_coin(coin)
        if room is None or room not in dist_map:
            return None
        coin_distances.append(dist_map[room])
    if not coin_distances:
        return None
    min_d = min(coin_distances)
    mean_d = sum(coin_distances) / len(coin_distances)
    min_ok = min_d >= 2
    mean_threshold = max(1.5, n * 0.25)
    mean_ok = mean_d >= mean_threshold
    if not (min_ok and mean_ok):
        return None
    return {"min_distance": min_d, "mean_distance": mean_d, "score": mean_d + 0.5 * min_d}


def sample_maps(num_locations, num_samples=100_000, num_coins=2):
    include_doors = True
    num_distractor_items = 2
    coins_in_containers = False
    rng = random.Random(42)
    connectivity_options = [0.08, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    candidates = []
    for _ in range(num_samples):
        seed = rng.randint(0, 2**31 - 1)
        connectivity = rng.choice(connectivity_options)
        game = CoinCollectorGame(
            num_locations=num_locations,
            num_distractor_items=num_distractor_items,
            include_doors=include_doors,
            num_coins=num_coins,
            connectivity=connectivity,
            coins_in_containers=coins_in_containers,
            seed=seed,
        )
        try:
            result = evaluate_map(game)
        except Exception:
            continue
        if result is not None:
            result["seed"] = seed
            result["connectivity"] = connectivity
            candidates.append(result)
    return candidates


def select_diverse(candidates, num_select, num_locations):
    if len(candidates) <= num_select:
        return sorted(candidates, key=lambda x: -x["score"])[:num_select]
    by_conn = {}
    for c in candidates:
        conn = c["connectivity"]
        if conn not in by_conn:
            by_conn[conn] = []
        by_conn[conn].append(c)
    for conn in by_conn:
        by_conn[conn].sort(key=lambda x: -x["score"])
    selected = []
    conn_keys = sorted(by_conn.keys())
    for _ in range(num_select):
        best = None
        best_conn = None
        for conn in conn_keys:
            if not by_conn[conn]:
                continue
            c = by_conn[conn][0]
            if best is None or c["score"] > best["score"]:
                best = c
                best_conn = conn
        if best is None:
            break
        selected.append(best)
        by_conn[best_conn].pop(0)
    return selected


def main():
    out_path = Path(__file__).resolve().parent / "dataset_selected.json"
    target_5 = 5
    target_10 = 10
    target_15 = 5
    n_samples = 100_000

    print("Sampling 5-room maps...")
    c5 = sample_maps(5, num_samples=n_samples)
    print(f"  {len(c5)} candidates")
    s5 = select_diverse(c5, target_5, 5)

    print("Sampling 10-room maps...")
    c10 = sample_maps(10, num_samples=n_samples)
    print(f"  {len(c10)} candidates")
    s10 = select_diverse(c10, target_10, 10)

    print("Sampling 15-room maps...")
    c15 = sample_maps(15, num_samples=n_samples)
    print(f"  {len(c15)} candidates")
    s15 = select_diverse(c15, target_15, 15)

    topologies_5 = ["linear", "straight line", "one cycle", "branching", "hub and spokes"]
    topologies_10 = [
        "linear", "straight line", "one cycle", "two cycles", "branchy path",
        "complex branching", "grid-like", "tree with cycles", "chain with side branches", "dense mesh"
    ]
    topologies_15 = ["linear", "long cycle", "multi-branch", "complex mesh", "highly connected"]

    maps = []
    for i, c in enumerate(s5):
        maps.append({
            "id": f"5room_{i+1:02d}",
            "num_locations": 5,
            "seed": c["seed"],
            "connectivity": c["connectivity"],
            "topology": topologies_5[i] if i < len(topologies_5) else "diverse",
        })
    for i, c in enumerate(s10):
        maps.append({
            "id": f"10room_{i+1:02d}",
            "num_locations": 10,
            "seed": c["seed"],
            "connectivity": c["connectivity"],
            "topology": topologies_10[i] if i < len(topologies_10) else "diverse",
        })
    for i, c in enumerate(s15):
        maps.append({
            "id": f"15room_{i+1:02d}",
            "num_locations": 15,
            "seed": c["seed"],
            "connectivity": c["connectivity"],
            "topology": topologies_15[i] if i < len(topologies_15) else "diverse",
        })

    with open(out_path, "w") as f:
        json.dump(maps, f, indent=2)
    print(f"Wrote {len(maps)} selected maps to {out_path}")
    print("Run dataset/scripts/build_dataset.py to regenerate dataset/maps from this list.")


if __name__ == "__main__":
    main()
