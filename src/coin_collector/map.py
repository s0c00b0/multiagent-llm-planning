"""Map generation for Coin Collector game."""

import random
import math
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
        fold: str = "train"
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
        
        map_result = self._mk_connections(r, locations)
        if map_result is None:
            raise RuntimeError("Could not generate connection map")
        
        self._connect_rooms_from_map(r, map_result, include_doors, locations, connectivity)
        
        available_locations = list(range(1, len(locations))) if len(locations) > 1 else []
        
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
    
    def _mk_connections(self, r: random.Random, locations: List[Room]) -> Optional[List[List[Optional[Room]]]]:
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

