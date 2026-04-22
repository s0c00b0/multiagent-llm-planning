"""Summarize experiment JSONs under a parent output folder.

Example:
    python3 src/summarize_output_folder.py outMessageQueue
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _find_game_jsons(parent_output_dir: Path) -> List[Path]:
    return sorted(parent_output_dir.glob("*/game_*.json"))


def _read_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _count_messages_for_run(game_data: Dict) -> int:
    shared_messages_final = game_data.get("shared_messages_final", [])
    if not isinstance(shared_messages_final, list):
        return 0
    return len(shared_messages_final)


def summarize(parent_output_dir: Path) -> int:
    if not parent_output_dir.exists():
        raise FileNotFoundError(f"Folder not found: {parent_output_dir}")
    if not parent_output_dir.is_dir():
        raise ValueError(f"Path is not a directory: {parent_output_dir}")

    json_paths = _find_game_jsons(parent_output_dir)
    if not json_paths:
        print(f"No game_*.json files found under: {parent_output_dir}")
        return 0

    for json_path in json_paths:
        data = _read_json(json_path)
        run_name = json_path.parent.name

        num_locations = data.get("num_locations")
        num_coins = data.get("num_coins")
        num_players = data.get("num_players")
        max_steps = data.get("max_steps")
        turns = data.get("total_turns")
        total_messages_sent = _count_messages_for_run(data)

        print(f"Run: {run_name}")
        print("Room Config:")
        print(f'  "num_locations": {num_locations},')
        print(f'  "num_coins": {num_coins},')
        print(f'  "num_players": {num_players},')
        print(f'  "max_steps": {max_steps},')
        print(f"Turns: {turns}")
        print(f"Total Messages Sent: {total_messages_sent}")
        print()

    return len(json_paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize room config, turns, and message counts from run JSONs."
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Parent output folder containing run subfolders (e.g., outMessageQueue)",
    )
    args = parser.parse_args()

    count = summarize(Path(args.output_folder))
    if count > 0:
        print(f"Processed {count} run(s).")


if __name__ == "__main__":
    main()
