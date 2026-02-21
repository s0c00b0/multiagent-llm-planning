"""Map generation for Coin Collector game."""

import random
import math
from collections import deque
from typing import Dict, List, Optional, Tuple

from .objects import Room, Door, FastObject, Container, Surface, Coin


class DoorMaker:
    def __init__(self):
        self.lut: Dict[str, List[str]] = {}
        self._init()
    
    def _mk_key(self, location1: str, location2: str) -> str:
        return f"{location1}+{location2}"
    
    def _add_to_lut(self, location1: str, location2: str, value: List[str]):
        key = self._mk_key(location1, location2)
        self.lut[key] = value
    
    def _init(self):
        self._add_to_lut("pantry", "kitchen", ["frosted-glass door", "plain door"])
        self._add_to_lut("kitchen", "backyard", ["sliding patio door", "patio door", "screen door"])
        self._add_to_lut("corridor", "backyard", ["sliding patio door", "patio door", "screen door"])
        self._add_to_lut("living room", "backyard", ["sliding patio door", "patio door", "screen door"])
        self._add_to_lut("living room", "driveway", ["front door", "fiberglass door"])
        self._add_to_lut("corridor", "driveway", ["front door", "fiberglass door"])
        self._add_to_lut("supermarket", "street", ["sliding door", "commercial glass door"])
        
        generic_doors = ["wood door"]
        for loc1, loc2 in [
            ("bedroom", "living room"), ("bedroom", "bathroom"), ("bedroom", "corridor"),
            ("bathroom", "living room"), ("bathroom", "corridor"), ("bathroom", "kitchen"),
            ("laundry room", "kitchen"), ("laundry room", "bathroom"), ("laundry room", "corridor")
        ]:
            self._add_to_lut(loc1, loc2, generic_doors)
    
    def mk_door(self, r: random.Random, location1: str, location2: str, is_open: bool) -> Optional[Door]:
        key1 = self._mk_key(location1, location2)
        key2 = self._mk_key(location2, location1)
        
        key = key1 if key1 in self.lut else key2
        if key not in self.lut:
            return None
        
        possible_doors = self.lut[key]
        description = r.choice(possible_doors)
        return Door(description, is_open)


class Kitchen(Room):
    def __init__(self, r: random.Random):
        super().__init__("kitchen", ["corridor", "pantry", "backyard", "living room"])
        self.add_object(Container("fridge", is_open=False, is_openable=True))
        self.add_object(Surface("counter"))
        self.add_object(Container("cutlery drawer", is_open=False, is_openable=True))
        self.add_object(Surface("dining table"))


class Pantry(Room):
    def __init__(self, r: random.Random):
        super().__init__("pantry", ["kitchen"])
        self.add_object(Surface("shelf"))


class Corridor(Room):
    def __init__(self, r: random.Random):
        super().__init__("corridor", ["kitchen", "bathroom", "backyard", "laundry room", "bedroom", "living room"])
        self.add_object(Surface("key holder"))
        self.add_object(Container("shoe cabinet", is_open=False, is_openable=True))


class Bedroom(Room):
    def __init__(self, r: random.Random):
        super().__init__("bedroom", ["corridor", "living room"])
        self.add_object(Surface("dressing table"))
        self.add_object(Surface("desk"))
        self.add_object(Container("chest of drawers", is_open=False, is_openable=True))
        self.add_object(Container("wardrobe", is_open=False, is_openable=True))
        self.add_object(Surface("bed"))


class Backyard(Room):
    def __init__(self, r: random.Random):
        super().__init__("backyard", ["corridor", "kitchen", "living room"])
        self.add_object(Surface("patio chair"))
        self.add_object(Surface("patio table"))


class LivingRoom(Room):
    def __init__(self, r: random.Random):
        super().__init__("living room", ["kitchen", "bathroom", "backyard", "bedroom"])
        self.add_object(Surface("sofa"))
        self.add_object(Surface("coffee table"))
        self.add_object(Surface("end table"))


class Bathroom(Room):
    def __init__(self, r: random.Random):
        super().__init__("bathroom", ["corridor", "kitchen", "bedroom", "living room"])
        self.add_object(Container("bathroom cabinet", is_open=False, is_openable=True))
        self.add_object(Surface("sink"))


class LaundryRoom(Room):
    def __init__(self, r: random.Random):
        super().__init__("laundry room", ["corridor", "kitchen", "bathroom"])
        self.add_object(Container("washing machine", is_open=False, is_openable=True))
        self.add_object(Surface("work table"))


class Driveway(Room):
    def __init__(self, r: random.Random):
        super().__init__("driveway", ["corridor", "backyard"])


class Street(Room):
    def __init__(self, r: random.Random):
        super().__init__("street", ["driveway", "backyard", "supermarket"])


class Supermarket(Room):
    def __init__(self, r: random.Random):
        super().__init__("supermarket", ["street"])
        self.add_object(Surface("showcase"))


class CoinGameGenerator:
    def __init__(self):
        self.door_maker = DoorMaker()
    
    def _create_room(self, r: random.Random, room_type: str) -> Room:
        room_classes = {
            "kitchen": Kitchen,
            "pantry": Pantry,
            "corridor": Corridor,
            "bedroom": Bedroom,
            "backyard": Backyard,
            "living room": LivingRoom,
            "bathroom": Bathroom,
            "laundry room": LaundryRoom,
            "driveway": Driveway,
            "street": Street,
            "supermarket": Supermarket,
        }
        return room_classes[room_type](r)
    
    def mk_environment(
        self,
        r: random.Random,
        num_locations: int,
        num_distractor_items: int,
        include_doors: bool,
        num_coins: int = 1,
        connectivity: float = 0.5,
        coins_in_containers: bool = False,
        fold: str = "train",
        topology: Optional[str] = None,
    ) -> Tuple[List[Room], List[FastObject]]:
        locations: List[Room] = []
        
        standard_room_types = [
            "kitchen", "pantry", "corridor", "bedroom", "backyard",
            "living room", "bathroom", "laundry room", "driveway",
            "street", "supermarket"
        ]
        
        room_types = []
        for i in range(num_locations):
            if i < len(standard_room_types):
                room_types.append(standard_room_types[i])
            else:
                base_room = standard_room_types[i % len(standard_room_types)]
                room_num = (i // len(standard_room_types)) + 1
                room_types.append(f"{base_room} {room_num}")
        
        for room_type in room_types:
            if room_type in standard_room_types:
                locations.append(self._create_room(r, room_type))
            else:
                base_type = room_type.split()[0]
                if base_type in standard_room_types:
                    base_room = self._create_room(r, base_type)
                    base_room.name = room_type
                    base_room.prefers_connecting_to = standard_room_types.copy()
                    locations.append(base_room)
                else:
                    generic_room = Room(room_type, standard_room_types.copy())
                    locations.append(generic_room)
        
        map_result = self._mk_connections(r, locations, topology)
        if map_result is None:
            raise RuntimeError("Could not generate connection map")
        
        self._connect_rooms_from_map(r, map_result, include_doors, locations, connectivity)

        available_locations = self._coin_eligible_locations(locations, num_coins)
        
        task_objects = []
        if available_locations and num_coins > 0:
            num_coins_to_place = min(num_coins, len(available_locations))
            coin_locations = r.sample(available_locations, num_coins_to_place)
            
            for i, coin_location_idx in enumerate(coin_locations):
                coin = Coin()
                if num_coins_to_place > 1:
                    coin.name = f"coin{i+1}"
                
                target_location = locations[coin_location_idx]
                if coins_in_containers:
                    containers = [obj for obj in target_location.contents if obj.is_container]
                    if containers:
                        container = r.choice(containers)
                        container.add_object(coin)
                    else:
                        target_location.add_object(coin)
                else:
                    target_location.add_object(coin)
                
                task_objects.append(coin)
        
        distractor_names = ["apple", "book", "pen", "key", "wallet", "phone", "watch", "glasses", "hat", "bag"]
        for _ in range(num_distractor_items):
            if locations:
                loc = r.choice(locations)
                containers = [obj for obj in loc.contents if obj.is_container]
                if containers:
                    container = r.choice(containers)
                    distractor = FastObject(r.choice(distractor_names))
                    distractor.is_movable = True
                    container.add_object(distractor)
                else:
                    distractor = FastObject(r.choice(distractor_names))
                    distractor.is_movable = True
                    loc.add_object(distractor)
        
        return locations, task_objects

    @staticmethod
    def _graph_distances_from(locations: List[Room], start_index: int) -> List[int]:
        """BFS from locations[start_index]. Returns list of length len(locations) with shortest step count (or -1 if unreachable)."""
        n = len(locations)
        room_to_idx = {loc: i for i, loc in enumerate(locations)}
        dist = [-1] * n
        dist[start_index] = 0
        q: deque = deque([start_index])
        while q:
            i = q.popleft()
            room = locations[i]
            for neighbor in (room.location_north, room.location_south, room.location_east, room.location_west):
                if neighbor is None:
                    continue
                j = room_to_idx.get(neighbor)
                if j is not None and dist[j] == -1:
                    dist[j] = dist[i] + 1
                    q.append(j)
        return dist

    def _coin_eligible_locations(self, locations: List[Room], num_coins: int) -> List[int]:
        """Indices of rooms where coins can be placed: exclude start (0), prefer at least 4 steps away so most of the map is useful."""
        if len(locations) <= 1 or num_coins <= 0:
            return []
        dist = self._graph_distances_from(locations, 0)
        for min_d in (4, 3, 2, 1):
            eligible = [i for i in range(1, len(locations)) if dist[i] >= min_d and dist[i] >= 0]
            if len(eligible) >= num_coins:
                return eligible
        return list(range(1, len(locations)))

    @staticmethod
    def _normalize_topology(topology: Optional[str]) -> Optional[str]:
        if not topology:
            return None
        t = topology.lower().strip()
        if t in ("linear", "straight line"):
            return "linear"
        if t in ("one cycle", "long cycle"):
            return "cycle"
        if t in ("hub and spokes", "star"):
            return "star"
        if t in ("branching", "branchy path", "complex branching", "multi-branch", "chain with side branches"):
            return "tree"
        if t in ("two cycles", "tree with cycles"):
            return "tree_with_cycles"
        if t in ("grid-like", "dense mesh", "complex mesh", "highly connected"):
            return "grid"
        return t

    @staticmethod
    def _placement_linear(n: int) -> List[Tuple[int, int]]:
        return [(0, c) for c in range(n)]

    @staticmethod
    def _placement_cycle(r: random.Random, n: int) -> List[Tuple[int, int]]:
        if n <= 2:
            return [(0, c) for c in range(n)]
        w = (n + 1) // 2
        positions: List[Tuple[int, int]] = []
        for c in range(w):
            positions.append((0, c))
        for c in range(w - 1, -1, -1):
            if len(positions) < n:
                positions.append((1, c))
        return positions[:n]

    @staticmethod
    def _placement_star(r: random.Random, n: int) -> List[Tuple[int, int]]:
        if n <= 1:
            return [(0, 0)] if n == 1 else []
        center = (0, 0)
        positions = [center]
        arms = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        r.shuffle(arms)
        arm_lengths = [0] * 4
        for _ in range(n - 1):
            i = min(range(4), key=lambda i: arm_lengths[i])
            arm_lengths[i] += 1
        for arm_idx, (dr, dc) in enumerate(arms):
            for step in range(1, arm_lengths[arm_idx] + 1):
                positions.append((dr * step, dc * step))
        return positions

    @staticmethod
    def _placement_tree(r: random.Random, n: int) -> List[Tuple[int, int]]:
        if n <= 0:
            return []
        used = {(0, 0)}
        positions = [(0, 0)]
        queue: deque[Tuple[int, int]] = deque([(0, 0)])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        while len(positions) < n and queue:
            ri, cj = queue.popleft()
            r.shuffle(directions)
            for dr, dc in directions:
                if len(positions) >= n:
                    break
                nr, nc = ri + dr, cj + dc
                if (nr, nc) not in used:
                    used.add((nr, nc))
                    positions.append((nr, nc))
                    queue.append((nr, nc))
        return positions

    @staticmethod
    def _placement_tree_with_cycles(r: random.Random, n: int) -> List[Tuple[int, int]]:
        if n <= 3:
            return CoinGameGenerator._placement_cycle(r, n)
        cycle_size = min(n, 4 + r.randint(0, max(0, (n - 4) // 2)))
        positions = list(CoinGameGenerator._placement_cycle(r, cycle_size))
        used = set(positions)
        branch_start = r.randint(0, len(positions) - 1)
        br, bc = positions[branch_start]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        r.shuffle(directions)
        for dr, dc in directions:
            nr, nc = br + dr, bc + dc
            if (nr, nc) not in used and len(positions) < n:
                used.add((nr, nc))
                positions.append((nr, nc))
                break
        queue: deque[Tuple[int, int]] = deque(positions[cycle_size:])
        while len(positions) < n and queue:
            ri, cj = queue.popleft()
            r.shuffle(directions)
            for dr, dc in directions:
                if len(positions) >= n:
                    break
                nr, nc = ri + dr, cj + dc
                if (nr, nc) not in used:
                    used.add((nr, nc))
                    positions.append((nr, nc))
                    queue.append((nr, nc))
        return positions

    @staticmethod
    def _placement_chain_with_side_branches(r: random.Random, n: int) -> List[Tuple[int, int]]:
        if n <= 2:
            return [(0, c) for c in range(n)]
        main_len = max(2, n - (n // 3))
        positions = [(0, c) for c in range(main_len)]
        used = set(positions)
        branch_nodes = r.sample(range(1, main_len), min(main_len - 1, n - main_len))
        for i, col in enumerate(branch_nodes):
            if len(positions) >= n:
                break
            side = 1 if (col % 2) == 0 else -1
            pos = (side, col)
            if pos not in used:
                used.add(pos)
                positions.append(pos)
        return positions[:n]

    @staticmethod
    def _placement_grid(r: random.Random, n: int) -> List[Tuple[int, int]]:
        if n <= 0:
            return []
        cols = max(1, int(math.ceil(math.sqrt(n))))
        rows = max(1, (n + cols - 1) // cols)
        positions = []
        for ri in range(rows):
            for cj in range(cols):
                if len(positions) < n:
                    positions.append((ri, cj))
        return positions

    def _topology_placement(self, r: random.Random, n: int, topology: str) -> List[Tuple[int, int]]:
        norm = self._normalize_topology(topology)
        if norm == "linear":
            return self._placement_linear(n)
        if norm == "cycle":
            return self._placement_cycle(r, n)
        if norm == "star":
            return self._placement_star(r, n)
        if norm == "tree":
            return self._placement_tree(r, n)
        if norm == "tree_with_cycles":
            return self._placement_tree_with_cycles(r, n)
        if norm == "grid":
            return self._placement_grid(r, n)
        if "chain" in topology.lower() and "branch" in topology.lower():
            return self._placement_chain_with_side_branches(r, n)
        return self._placement_tree(r, n)

    def _mk_connections(self, r: random.Random, locations: List[Room], topology: Optional[str] = None) -> Optional[List[List[Optional[Room]]]]:
        n = len(locations)
        if topology:
            positions = self._topology_placement(r, n, topology)
            if len(positions) != n:
                positions = self._placement_linear(n)
            rows = [p[0] for p in positions]
            cols = [p[1] for p in positions]
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            GRID_SIZE = max(max_r - min_r + 3, max_c - min_c + 3, 7)
            offset_r = (GRID_SIZE - (max_r - min_r + 1)) // 2 - min_r
            offset_c = (GRID_SIZE - (max_c - min_c + 1)) // 2 - min_c
            map_grid = [[None] * GRID_SIZE for _ in range(GRID_SIZE)]
            indices = list(range(n))
            r.shuffle(indices)
            for i in range(n):
                loc = locations[indices[i]]
                pr, pc = positions[i]
                gr, gc = pr + offset_r, pc + offset_c
                if 0 <= gr < GRID_SIZE and 0 <= gc < GRID_SIZE:
                    map_grid[gr][gc] = loc
            return map_grid

        min_grid_size = int(math.ceil(math.sqrt(len(locations) * 1.5))) + 2
        GRID_SIZE = max(7, min_grid_size)
        map_grid = [[None] * GRID_SIZE for _ in range(GRID_SIZE)]
        locations_left = locations.copy()
        r.shuffle(locations_left)
        center = GRID_SIZE // 2
        last_x, last_y = center, center
        map_grid[last_x][last_y] = locations_left.pop()
        populated = [(last_x, last_y)]
        attempts = 0
        max_attempts = len(locations) * 20
        while locations_left and attempts < max_attempts:
            ref_idx = r.randint(0, len(populated) - 1)
            ref_x, ref_y = populated[ref_idx]
            last_location = map_grid[ref_x][ref_y]
            location_idx = r.randint(0, len(locations_left) - 1)
            location = locations_left[location_idx]
            prefers_connection = (
                last_location.name in location.prefers_connecting_to or
                location.name in last_location.prefers_connecting_to
            )
            lenient_connection = (
                not location.prefers_connecting_to or
                not last_location.prefers_connecting_to
            )
            force_connection = attempts > max_attempts * 0.7 and len(locations_left) > 0
            if prefers_connection or lenient_connection or force_connection:
                new_x, new_y = self._find_empty_direction(r, map_grid, ref_x, ref_y)
                if new_x != -1:
                    map_grid[new_x][new_y] = location
                    locations_left.pop(location_idx)
                    populated.append((new_x, new_y))
            attempts += 1
        if locations_left:
            return None
        return map_grid
    
    def _find_empty_direction(self, r: random.Random, map_grid: List[List[Optional[Room]]], loc_x: int, loc_y: int) -> Tuple[int, int]:
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        r.shuffle(directions)
        
        for dx, dy in directions:
            new_x, new_y = loc_x + dx, loc_y + dy
            if 0 <= new_x < len(map_grid) and 0 <= new_y < len(map_grid[0]):
                if map_grid[new_x][new_y] is None:
                    return new_x, new_y
        
        return -1, -1
    
    def _connect_rooms_from_map(self, r: random.Random, map_grid: List[List[Optional[Room]]], include_doors: bool, locations: List[Room], connectivity: float = 0.5):
        possible_connections = []
        
        for i in range(len(map_grid)):
            for j in range(len(map_grid[i])):
                cell = map_grid[i][j]
                if cell is None:
                    continue
                
                if i < len(map_grid) - 1:
                    query_loc = map_grid[i + 1][j]
                    if query_loc is not None:
                        prefers = (
                            query_loc.name in cell.prefers_connecting_to or
                            cell.name in query_loc.prefers_connecting_to
                        )
                        possible_connections.append(('north', i, j, cell, query_loc, prefers))
                
                if i >= 1:
                    query_loc = map_grid[i - 1][j]
                    if query_loc is not None:
                        prefers = (
                            query_loc.name in cell.prefers_connecting_to or
                            cell.name in query_loc.prefers_connecting_to
                        )
                        possible_connections.append(('south', i, j, cell, query_loc, prefers))
                
                if j >= 1:
                    query_loc = map_grid[i][j - 1]
                    if query_loc is not None:
                        prefers = (
                            query_loc.name in cell.prefers_connecting_to or
                            cell.name in query_loc.prefers_connecting_to
                        )
                        possible_connections.append(('east', i, j, cell, query_loc, prefers))
                
                if j < len(map_grid[i]) - 1:
                    query_loc = map_grid[i][j + 1]
                    if query_loc is not None:
                        prefers = (
                            query_loc.name in cell.prefers_connecting_to or
                            cell.name in query_loc.prefers_connecting_to
                        )
                        possible_connections.append(('west', i, j, cell, query_loc, prefers))
        
        parent = {loc.name: loc.name for loc in locations}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            root_x, root_y = find(x), find(y)
            if root_x != root_y:
                parent[root_y] = root_x
                return True
            return False
        
        preferred_connections = [c for c in possible_connections if c[5]]
        other_connections = [c for c in possible_connections if not c[5]]
        r.shuffle(preferred_connections)
        r.shuffle(other_connections)
        
        spanning_tree_edges = []
        for direction, i, j, cell, query_loc, prefers in preferred_connections:
            if union(cell.name, query_loc.name):
                spanning_tree_edges.append((direction, i, j, cell, query_loc, prefers))
        
        for direction, i, j, cell, query_loc, prefers in other_connections:
            if union(cell.name, query_loc.name):
                spanning_tree_edges.append((direction, i, j, cell, query_loc, prefers))
        
        all_unique_possible_edges = set()
        for direction, i, j, cell, query_loc, prefers in possible_connections:
            edge_key = tuple(sorted([cell.name, query_loc.name]))
            all_unique_possible_edges.add(edge_key)
        
        min_edges = len(spanning_tree_edges)
        max_edges = len(all_unique_possible_edges)
        
        target_edges = int(min_edges + connectivity * (max_edges - min_edges))
        
        all_edges = preferred_connections + other_connections
        r.shuffle(all_edges)
        
        edges_to_add = set()
        for direction, i, j, cell, query_loc, prefers in spanning_tree_edges:
            edge_key = tuple(sorted([cell.name, query_loc.name]))
            edges_to_add.add(edge_key)
        
        for direction, i, j, cell, query_loc, prefers in all_edges:
            if len(edges_to_add) >= target_edges:
                break
            edge_key = tuple(sorted([cell.name, query_loc.name]))
            if edge_key not in edges_to_add:
                edges_to_add.add(edge_key)
        
        for direction, i, j, cell, query_loc, prefers in all_edges:
            edge_key = tuple(sorted([cell.name, query_loc.name]))
            if edge_key in edges_to_add:
                if direction == 'north':
                    cell.location_north = query_loc
                    query_loc.location_south = cell
                    if include_doors:
                        door = self.door_maker.mk_door(r, cell.name, query_loc.name, is_open=False)
                        if door is not None:
                            cell.door_north = door
                            query_loc.door_south = door
                elif direction == 'south':
                    cell.location_south = query_loc
                    query_loc.location_north = cell
                    if include_doors:
                        door = self.door_maker.mk_door(r, cell.name, query_loc.name, is_open=False)
                        if door is not None:
                            cell.door_south = door
                            query_loc.door_north = door
                elif direction == 'east':
                    cell.location_east = query_loc
                    query_loc.location_west = cell
                    if include_doors:
                        door = self.door_maker.mk_door(r, cell.name, query_loc.name, is_open=False)
                        if door is not None:
                            cell.door_east = door
                            query_loc.door_west = door
                elif direction == 'west':
                    cell.location_west = query_loc
                    query_loc.location_east = cell
                    if include_doors:
                        door = self.door_maker.mk_door(r, cell.name, query_loc.name, is_open=False)
                        if door is not None:
                            cell.door_west = door
                            query_loc.door_east = door

