"""Coin Collector game implementation."""

import random
from typing import Dict, List, Set, Optional, Tuple, Any
from enum import Enum
from collections import deque

from .objects import FastObject, Door, Container, Surface, Coin, Room
from .map import CoinGameGenerator


class Direction(Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


class CoinGameScoring:
    def __init__(self, task_objects: List[FastObject]):
        self.task_objects = task_objects
        self.max_score = len(task_objects)
        self.cur_score = 0.0
        self.task_success = False
        self.task_failure = False
    
    def do_scoring(self, player_inventories: List[FastObject]):
        cur_score = 0.0
        
        for player_id, inventory in enumerate(player_inventories):
            for task_obj in self.task_objects:
                if task_obj.current_container is not None and task_obj.current_container.name == f"inventory_player_{player_id}":
                    cur_score += 1.0
        
        self.cur_score = cur_score
        self.task_success = (cur_score == self.max_score)
        self.task_failure = False


class CoinCollectorGame:
    def __init__(
        self,
        num_locations: int = 11,
        num_distractor_items: int = 0,
        include_doors: bool = True,
        limit_inventory_size: bool = True,
        num_players: int = 1,
        max_steps: Optional[int] = None,
        num_coins: int = 1,
        connectivity: float = 0.5,
        coins_in_containers: bool = False,
        seed: Optional[int] = None
    ):
        """
        Args:
            connectivity: 0.0 = minimal connections, 1.0 = maximum connections
            coins_in_containers: If True, coins may be hidden in containers
        """
        if num_locations < 1:
            raise ValueError(f"num_locations must be at least 1, got {num_locations}")
        if num_players < 1:
            raise ValueError(f"num_players must be at least 1, got {num_players}")
        if num_coins < 0:
            raise ValueError(f"num_coins must be non-negative, got {num_coins}")
        if num_distractor_items < 0:
            raise ValueError(f"num_distractor_items must be non-negative, got {num_distractor_items}")
        if connectivity < 0.0 or connectivity > 1.0:
            raise ValueError(f"connectivity must be between 0.0 and 1.0, got {connectivity}")
        if max_steps is not None and max_steps < 0:
            raise ValueError(f"max_steps must be non-negative, got {max_steps}")
        
        self.num_locations = num_locations
        self.num_distractor_items = num_distractor_items
        self.include_doors = include_doors
        self.limit_inventory_size = limit_inventory_size
        self.num_players = num_players
        self.max_steps = max_steps
        self.num_coins = num_coins
        self.connectivity = connectivity
        self.coins_in_containers = coins_in_containers
        self.seed = seed
        
        self.generator = CoinGameGenerator()
        self.locations: List[Room] = []
        self.task_objects: List[FastObject] = []
        self.player_locations: List[Optional[Room]] = []
        self.player_inventories: List[FastObject] = []
        self.current_player = 0
        self.player_steps: List[int] = []
        self.room_movement_events: Dict[Room, List[Tuple[int, str, str, Set[int]]]] = {}
        self.scorer: Optional[CoinGameScoring] = None
        self.last_valid_actions: List[Tuple[str, int, List[FastObject]]] = []
        self._random: Optional[random.Random] = None
        self.generation_properties = {
            "numLocations": num_locations,
            "numDistractorItems": num_distractor_items,
            "includeDoors": 1 if include_doors else 0,
            "limitInventorySize": 1 if limit_inventory_size else 0,
            "numPlayers": num_players,
            "maxSteps": max_steps if max_steps is not None else -1,
            "numCoins": num_coins,
            "connectivity": connectivity,
        }
        
        if seed is not None:
            self.generation_properties["seed"] = seed
    
    def reset(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            self.seed = seed
            self.generation_properties["seed"] = seed
        
        r = random.Random(self.seed if self.seed is not None else None)
        self._random = r
        
        self.locations, self.task_objects = self.generator.mk_environment(
            r, self.num_locations, self.num_distractor_items, self.include_doors, self.num_coins, self.connectivity, self.coins_in_containers
        )
        
        self.player_locations = [self.locations[0] for _ in range(self.num_players)]
        self.player_inventories = [FastObject(f"inventory_player_{i}") for i in range(self.num_players)]
        self.current_player = 0
        self.player_steps = [0] * self.num_players
        self.room_movement_events = {}
        self.scorer = CoinGameScoring(self.task_objects)
        self.scorer.do_scoring(self.player_inventories)
        self._generate_valid_actions(self.current_player)
        return self._get_observation(self.current_player)
    
    def _generate_valid_actions(self, player_id: int):
        if player_id < 0 or player_id >= len(self.player_locations):
            self.last_valid_actions = []
            return
        
        actions = []
        player_location = self.player_locations[player_id]
        
        if player_location is None:
            self.last_valid_actions = []
            return
        
        actions.append(("inventory", 14, []))
        if player_location.location_north is not None:
            door = player_location.door_north
            actions.append(("move north", 11, [door, player_location.location_north]))
        if player_location.location_south is not None:
            door = player_location.door_south
            actions.append(("move south", 11, [door, player_location.location_south]))
        if player_location.location_east is not None:
            door = player_location.door_east
            actions.append(("move east", 11, [door, player_location.location_east]))
        if player_location.location_west is not None:
            door = player_location.door_west
            actions.append(("move west", 11, [door, player_location.location_west]))
        if player_location.door_north is not None:
            actions.append(("open door to north", 16, [player_location.door_north, player_location.location_north]))
            actions.append(("close door to north", 17, [player_location.door_north, player_location.location_north]))
        if player_location.door_south is not None:
            actions.append(("open door to south", 16, [player_location.door_south, player_location.location_south]))
            actions.append(("close door to south", 17, [player_location.door_south, player_location.location_south]))
        if player_location.door_east is not None:
            actions.append(("open door to east", 16, [player_location.door_east, player_location.location_east]))
            actions.append(("close door to east", 17, [player_location.door_east, player_location.location_east]))
        if player_location.door_west is not None:
            actions.append(("open door to west", 16, [player_location.door_west, player_location.location_west]))
            actions.append(("close door to west", 17, [player_location.door_west, player_location.location_west]))
        
        visible_objects = player_location.collect_visible_objects()
        for obj in visible_objects:
            if obj.is_movable:
                actions.append((f"take {obj.name}", 1, [obj]))
            if obj.is_openable:
                if obj.is_open:
                    actions.append((f"close {obj.name}", 4, [obj]))
                else:
                    actions.append((f"open {obj.name}", 3, [obj]))
        
        if self._random is not None:
            self._random.shuffle(actions)
        else:
            random.shuffle(actions)
        self.last_valid_actions = actions
    
    def _get_observation(self, player_id: int) -> str:
        if player_id < 0 or player_id >= len(self.player_locations):
            return f"Invalid player_id: {player_id}"
        
        player_location = self.player_locations[player_id]
        if player_location is None:
            return f"Player {player_id + 1} has no location assigned."
        
        obs_parts = []
        
        if self.num_players > 1:
            obs_parts.append(f"[Player {self.current_player + 1}'s turn]\n")
        
        obs_parts.append(player_location.get_description())
        
        if self.num_players > 1:
            other_players_in_room = []
            for other_player_id, other_location in enumerate(self.player_locations):
                if other_player_id != player_id and other_location == player_location:
                    other_players_in_room.append(other_player_id + 1)
            
            if other_players_in_room:
                if len(other_players_in_room) == 1:
                    obs_parts.append(f"\nPlayer {other_players_in_room[0]} is also in this room.")
                else:
                    players_str = ", ".join([f"Player {p}" for p in other_players_in_room[:-1]])
                    obs_parts.append(f"\n{players_str}, and Player {other_players_in_room[-1]} are also in this room.")
            
            movement_events = self.room_movement_events.get(player_location, [])
            if movement_events:
                players_in_room = set()
                for other_player_id, other_location in enumerate(self.player_locations):
                    if other_location == player_location:
                        players_in_room.add(other_player_id)
                
                event_descriptions = []
                remaining_events = []
                
                for event_player_id, direction, event_type, seen_by in movement_events:
                    if event_player_id != player_id:
                        if player_id not in seen_by:
                            if event_type == 'entered':
                                event_descriptions.append(f"Player {event_player_id + 1} entered from the {direction}")
                            elif event_type == 'exited':
                                event_descriptions.append(f"Player {event_player_id + 1} exited to the {direction}")
                            seen_by.add(player_id)
                        
                        if not seen_by.issuperset(players_in_room):
                            remaining_events.append((event_player_id, direction, event_type, seen_by))
                    else:
                        remaining_events.append((event_player_id, direction, event_type, seen_by))
                
                if event_descriptions:
                    obs_parts.append("\n" + ". ".join(event_descriptions) + ".")
                
                if remaining_events:
                    self.room_movement_events[player_location] = remaining_events
                else:
                    if player_location in self.room_movement_events:
                        del self.room_movement_events[player_location]
        
        return "".join(obs_parts)
    
    def step(self, action: str, player_id: Optional[int] = None) -> Tuple[str, float, bool, Dict[str, Any]]:
        if self.scorer is None:
            return "Game has not been reset. Please call reset() first.", 0.0, False, {
                "validActions": [],
                "scoreRaw": 0.0,
                "scoreNormalized": 0.0,
                "taskSuccess": False,
                "taskFailure": False,
                "wasValidAction": False,
                "currentPlayer": 0,
                "playerSteps": []
            }
        
        if player_id is None:
            player_id = self.current_player
        
        if player_id < 0 or player_id >= self.num_players:
            return f"Invalid player_id: {player_id}. Must be between 0 and {self.num_players - 1}.", 0.0, False, {
                "validActions": [],
                "scoreRaw": self.scorer.cur_score,
                "scoreNormalized": self.scorer.cur_score / self.scorer.max_score if self.scorer.max_score > 0 else 0.0,
                "taskSuccess": self.scorer.task_success,
                "taskFailure": self.scorer.task_failure,
                "wasValidAction": False,
                "currentPlayer": self.current_player,
                "playerSteps": self.player_steps.copy()
            }
        
        if self.scorer.task_success:
            info = {
                "validActions": [],
                "scoreRaw": self.scorer.cur_score,
                "scoreNormalized": self.scorer.cur_score / self.scorer.max_score if self.scorer.max_score > 0 else 0.0,
                "taskSuccess": self.scorer.task_success,
                "taskFailure": self.scorer.task_failure,
                "wasValidAction": False,
                "currentPlayer": self.current_player,
                "playerSteps": self.player_steps.copy()
            }
            return "Game is already over. Please reset.", 0.0, True, info
        
        if self.max_steps is not None and self.player_steps[player_id] >= self.max_steps:
            info = {
                "validActions": [],
                "scoreRaw": self.scorer.cur_score,
                "scoreNormalized": self.scorer.cur_score / self.scorer.max_score if self.scorer.max_score > 0 else 0.0,
                "taskSuccess": self.scorer.task_success,
                "taskFailure": self.scorer.task_failure,
                "wasValidAction": False,
                "currentPlayer": self.current_player,
                "playerSteps": self.player_steps.copy()
            }
            return f"Player {player_id + 1} has reached maximum steps ({self.max_steps}). Game over.", 0.0, True, info
        
        action_found = None
        for act_str, act_idx, params in self.last_valid_actions:
            if act_str == action:
                action_found = (act_str, act_idx, params)
                break
        
        if action_found is None:
            observation = "That is not a command that I recognize."
            reward = 0.0
            done = False
            was_valid = False
        else:
            _, act_idx, params = action_found
            observation = self._run_action(act_idx, params, player_id)
            was_valid = True
            self.player_steps[player_id] += 1
        
        self.scorer.do_scoring(self.player_inventories)
        done = self.scorer.task_success
        reward = 1.0 if done else 0.0
        
        if not done:
            if player_id is None:
                self.current_player = (self.current_player + 1) % self.num_players
            else:
                self.current_player = (player_id + 1) % self.num_players
            self._generate_valid_actions(self.current_player)
        else:
            self._generate_valid_actions(player_id)
        
        info = {
            "validActions": [act[0] for act in self.last_valid_actions],
            "scoreRaw": self.scorer.cur_score,
            "scoreNormalized": self.scorer.cur_score / self.scorer.max_score if self.scorer.max_score > 0 else 0.0,
            "taskSuccess": self.scorer.task_success,
            "taskFailure": self.scorer.task_failure,
            "wasValidAction": was_valid,
            "currentPlayer": self.current_player,
            "playerSteps": self.player_steps.copy()
        }
        
        return observation, reward, done, info
    
    def _run_action(self, action_idx: int, params: List[FastObject], player_id: int) -> str:
        ACTION_TAKE = 1
        ACTION_OPEN = 3
        ACTION_CLOSE = 4
        ACTION_MOVE = 11
        ACTION_INVENTORY = 14
        ACTION_OPENDOOR = 16
        ACTION_CLOSEDOOR = 17
        
        if player_id < 0 or player_id >= len(self.player_locations):
            return f"Invalid player_id: {player_id}"
        
        player_location = self.player_locations[player_id]
        player_inventory = self.player_inventories[player_id]
        
        if player_location is None:
            return f"Player {player_id + 1} has no location assigned."
        if player_inventory is None:
            return f"Player {player_id + 1} has no inventory assigned."
        
        if action_idx == ACTION_INVENTORY:
            if self.limit_inventory_size:
                max_capacity = len(self.task_objects) + 1
                inv_str = f"Player {player_id + 1}'s Inventory (maximum capacity is {max_capacity} items): \n"
            else:
                inv_str = f"Player {player_id + 1}'s Inventory: \n"
            
            if not player_inventory.contents:
                inv_str += "  Your inventory is currently empty.\n"
            else:
                for obj in player_inventory.contents:
                    inv_str += f"  {obj.get_description()}\n"
            return inv_str
        
        elif action_idx == ACTION_TAKE:
            if not params or len(params) == 0:
                return "No object specified to take."
            obj = params[0]
            if obj is None:
                return "Invalid object to take."
            
            if self.limit_inventory_size:
                max_capacity = len(self.task_objects) + 1
                if len(player_inventory.contents) >= max_capacity:
                    return f"Your inventory currently has {len(player_inventory.contents)} items, and is full. You can't pick up another item."
            
            obj.remove_from_current_container()
            player_inventory.add_object(obj)
            return f"Player {player_id + 1} takes the {obj.name}."
        
        elif action_idx == ACTION_OPEN:
            if not params or len(params) == 0:
                return "No object specified to open."
            obj = params[0]
            if obj is None:
                return "Invalid object to open."
            if obj.is_open:
                return "That is already open."
            obj.is_open = True
            if not obj.contents:
                return f"You open the {obj.name}. It's empty inside."
            items = ", ".join([item.get_description() for item in obj.contents])
            return f"You open the {obj.name}. The {obj.name} contains {items}."
        
        elif action_idx == ACTION_CLOSE:
            if not params or len(params) == 0:
                return "No object specified to close."
            obj = params[0]
            if obj is None:
                return "Invalid object to close."
            if not obj.is_open:
                return "That is already closed."
            obj.is_open = False
            return f"You close the {obj.name}."
        
        elif action_idx == ACTION_MOVE:
            if not params or len(params) < 2:
                return "Invalid move parameters."
            door, new_loc = params[0], params[1]
            if new_loc is None:
                return "Invalid destination location."
            if door is not None and not door.is_open:
                return "You can't move there, the door is closed."
            
            old_loc = self.player_locations[player_id]
            
            exit_direction = None
            enter_direction = None
            if old_loc is not None:
                if old_loc.location_north == new_loc:
                    exit_direction = "north"
                    enter_direction = "south"
                elif old_loc.location_south == new_loc:
                    exit_direction = "south"
                    enter_direction = "north"
                elif old_loc.location_east == new_loc:
                    exit_direction = "east"
                    enter_direction = "west"
                elif old_loc.location_west == new_loc:
                    exit_direction = "west"
                    enter_direction = "east"
            
            # Record movement events for multiplayer visibility
            if self.num_players > 1 and old_loc is not None and exit_direction is not None:
                # Record exit event in old room (with empty seen_by set)
                if old_loc not in self.room_movement_events:
                    self.room_movement_events[old_loc] = []
                self.room_movement_events[old_loc].append((player_id, exit_direction, 'exited', set()))
                
                # Record enter event in new room (with direction they came from, empty seen_by set)
                if new_loc not in self.room_movement_events:
                    self.room_movement_events[new_loc] = []
                self.room_movement_events[new_loc].append((player_id, enter_direction, 'entered', set()))
            
            self.player_locations[player_id] = new_loc
            return self._get_observation(player_id)
        
        elif action_idx == ACTION_OPENDOOR:
            if not params or len(params) < 2:
                return "Invalid door parameters."
            door, location_beyond = params[0], params[1]
            if door is None:
                return "Invalid door."
            if door.is_open:
                return "That is already open."
            door.is_open = True
            return f"You open the {door.get_description()}, revealing the {location_beyond.name}."
        
        elif action_idx == ACTION_CLOSEDOOR:
            if not params or len(params) < 2:
                return "Invalid door parameters."
            door, location_beyond = params[0], params[1]
            if door is None:
                return "Invalid door."
            if not door.is_open:
                return "That is already closed."
            door.is_open = False
            return f"You close the {door.get_description()} to the {location_beyond.name}."
        
        else:
            return "That is not a command that I recognize."
    
    def get_observation(self, player_id: Optional[int] = None) -> str:
        if player_id is None:
            player_id = self.current_player
        return self._get_observation(player_id)
    
    def get_current_player(self) -> int:
        return self.current_player
    
    def get_generation_properties(self) -> Dict[str, int]:
        return self.generation_properties.copy()
    
    def get_gold_action_sequence(self) -> List[str]:
        return []
