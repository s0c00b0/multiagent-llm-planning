"""Compute the optimal (minimum) number of turns for a single player to collect all coins.

Uses BFS over the state space:
    state = (room_idx, coins_mask, doors_mask, containers_mask)

All four components are plain integers, so each state is a tuple of ints — cheap to hash
and compare. All actions cost exactly 1 turn, so BFS is guaranteed to find the optimum.

Key design choices:
- Doors and containers are only ever opened, never closed.  Closing either would strictly
  increase the cost, so the optimal path never does it.
- Containers are only opened when they contain at least one uncollected coin.
- Door/container state is tracked with bitmasks (one bit per object), capped at 64 objects
  each (far more than any generated map uses).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.coin_collector import CoinCollectorGame


def compute_optimal_turns(game: "CoinCollectorGame") -> Optional[int]:
    """Return the minimum number of single-player turns to collect all coins, or None
    if the goal is unreachable (e.g. disconnected map).

    The computation is performed on a snapshot of the game state immediately after
    reset(), so it must be called before any steps are taken.
    """
    rooms = game.locations
    coins = game.task_objects

    if not coins:
        return 0

    num_coins = len(coins)
    num_rooms = len(rooms)

    # ------------------------------------------------------------------ rooms
    room_to_idx: dict[int, int] = {id(r): i for i, r in enumerate(rooms)}

    # ------------------------------------------------------------------ doors
    # Enumerate every unique Door object reachable from any room.
    doors: list = []
    door_to_bit: dict[int, int] = {}
    for room in rooms:
        for attr in ("door_north", "door_south", "door_east", "door_west"):
            door = getattr(room, attr)
            if door is not None and id(door) not in door_to_bit:
                door_to_bit[id(door)] = 1 << len(doors)
                doors.append(door)

    initial_doors_mask: int = 0
    for i, d in enumerate(doors):
        if d.is_open:
            initial_doors_mask |= 1 << i

    # ------------------------------------------------------------------ containers
    # Only track openable containers (is_openable=True).  Surfaces are always open and
    # need no tracking.
    containers: list = []
    container_to_bit: dict[int, int] = {}
    for room in rooms:
        for obj in room.contents:
            if obj.is_container and obj.is_openable and id(obj) not in container_to_bit:
                container_to_bit[id(obj)] = 1 << len(containers)
                containers.append(obj)

    initial_containers_mask: int = 0
    for i, c in enumerate(containers):
        if c.is_open:
            initial_containers_mask |= 1 << i

    # ------------------------------------------------------------------ coin info
    # For each coin: (room_idx, container_bit)
    # container_bit == 0  →  coin is freely visible (on floor or on a surface)
    # container_bit != 0  →  must open that container first
    coin_info: list[tuple[Optional[int], int]] = []
    for coin in coins:
        container = coin.current_container
        if container is None:
            coin_info.append((None, 0))
            continue

        if id(container) in room_to_idx:
            # Directly in a room (placed on the floor)
            coin_info.append((room_to_idx[id(container)], 0))
        else:
            # In a container or on a surface inside a room
            cont_bit = container_to_bit.get(id(container), 0)  # 0 for surfaces
            parent = container.current_container
            if parent is not None and id(parent) in room_to_idx:
                coin_info.append((room_to_idx[id(parent)], cont_bit))
            else:
                # Nested more than one level — treat as unreachable for safety
                coin_info.append((None, 0))

    # ------------------------------------------------------------------ room adjacency
    # For each room: list of (neighbor_room_idx, door_bit)
    # door_bit == 0 means the passage is always open (no door).
    DIRECTION_PAIRS = (
        ("location_north", "door_north"),
        ("location_south", "door_south"),
        ("location_east",  "door_east"),
        ("location_west",  "door_west"),
    )
    room_neighbors: list[list[tuple[int, int]]] = []
    for room in rooms:
        nbrs: list[tuple[int, int]] = []
        for loc_attr, door_attr in DIRECTION_PAIRS:
            neighbor = getattr(room, loc_attr)
            if neighbor is None:
                continue
            neighbor_idx = room_to_idx[id(neighbor)]
            door = getattr(room, door_attr)
            door_bit = door_to_bit.get(id(door), 0) if door is not None else 0
            nbrs.append((neighbor_idx, door_bit))
        room_neighbors.append(nbrs)

    # ------------------------------------------------------------------ precompute per-room coin list
    # For each room: list of (coin_idx, container_bit)
    coins_in_room: list[list[tuple[int, int]]] = [[] for _ in range(num_rooms)]
    for ci, (crm, cbit) in enumerate(coin_info):
        if crm is not None:
            coins_in_room[crm].append((ci, cbit))

    # ------------------------------------------------------------------ precompute per-room container list
    # For each room: list of container_bit values for openable containers that hold at
    # least one coin.  Only these are ever worth opening.
    useful_containers_in_room: list[list[int]] = [[] for _ in range(num_rooms)]
    for room_idx, room in enumerate(rooms):
        for obj in room.contents:
            if not (obj.is_container and obj.is_openable):
                continue
            cbit = container_to_bit.get(id(obj), 0)
            if cbit == 0:
                continue
            # Worth opening only if it holds at least one coin
            if any(ci_bit == cbit for (_, ci_bit) in coins_in_room[room_idx]):
                useful_containers_in_room[room_idx].append(cbit)

    # ------------------------------------------------------------------ BFS
    start_room = game.player_locations[0]
    if start_room is None:
        return None
    start_room_idx = room_to_idx[id(start_room)]

    goal_coins_mask: int = (1 << num_coins) - 1

    # State: (room_idx, coins_mask, doors_mask, containers_mask)
    initial_state = (start_room_idx, 0, initial_doors_mask, initial_containers_mask)

    queue: deque[tuple[tuple[int, int, int, int], int]] = deque()
    queue.append((initial_state, 0))
    visited: set[tuple[int, int, int, int]] = {initial_state}

    while queue:
        state, turns = queue.popleft()
        room_idx, coins_mask, doors_mask, conts_mask = state

        if coins_mask == goal_coins_mask:
            return turns

        next_turns = turns + 1

        # -- Collect coins visible in this room -------------------------------
        for ci, cbit in coins_in_room[room_idx]:
            if coins_mask & (1 << ci):
                continue  # already collected
            # Accessible if no container needed, or container is already open
            if cbit == 0 or (conts_mask & cbit):
                ns = (room_idx, coins_mask | (1 << ci), doors_mask, conts_mask)
                if ns not in visited:
                    visited.add(ns)
                    queue.append((ns, next_turns))

        # -- Open a container in this room ------------------------------------
        for cbit in useful_containers_in_room[room_idx]:
            if conts_mask & cbit:
                continue  # already open
            # Only open if the container still has an uncollected coin
            if any(
                ci_bit == cbit and not (coins_mask & (1 << ci))
                for ci, ci_bit in coins_in_room[room_idx]
            ):
                ns = (room_idx, coins_mask, doors_mask, conts_mask | cbit)
                if ns not in visited:
                    visited.add(ns)
                    queue.append((ns, next_turns))

        # -- Open a door in this room -----------------------------------------
        for neighbor_idx, door_bit in room_neighbors[room_idx]:
            if door_bit and not (doors_mask & door_bit):
                ns = (room_idx, coins_mask, doors_mask | door_bit, conts_mask)
                if ns not in visited:
                    visited.add(ns)
                    queue.append((ns, next_turns))

        # -- Move to a neighboring room ---------------------------------------
        for neighbor_idx, door_bit in room_neighbors[room_idx]:
            if door_bit == 0 or (doors_mask & door_bit):
                ns = (neighbor_idx, coins_mask, doors_mask, conts_mask)
                if ns not in visited:
                    visited.add(ns)
                    queue.append((ns, next_turns))

    return None  # Goal unreachable
