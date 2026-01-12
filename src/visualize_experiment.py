"""Visualize a single experiment run with player paths on the map."""

import json
import argparse
import sys
import os
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path as MPLPath

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.coin_collector.game import CoinCollectorGame
from src.coin_collector.visualize import visualize_map, _calculate_layout


def load_experiment(experiment_dir: str) -> Tuple[Dict, Path]:
    path = Path(experiment_dir)
    
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {experiment_dir}")
    
    json_files = list(path.glob("game_*.json"))
    if not json_files:
        raise ValueError(f"No game_*.json file found in {experiment_dir}")
    json_path = json_files[0]
    
    with open(json_path, 'r') as f:
        return json.load(f), path


def simulate_player_paths(
    game: CoinCollectorGame,
    player_actions: List[List[str]],
    num_players: int
) -> Tuple[List[List[Tuple[str, int]]], Dict[Tuple[int, str], bool]]:
    game.reset()
    
    player_paths = [[] for _ in range(num_players)]
    room_actions = {}
    
    for player_id in range(num_players):
        initial_room = game.player_locations[player_id]
        if initial_room:
            player_paths[player_id].append((initial_room.name, 0))
    
    def is_movement_action(action: str) -> bool:
        return action.startswith('move ')
    
    action_indices = [0] * num_players
    turn = 0
    max_actions = max(len(actions) for actions in player_actions)
    
    while turn < max_actions * num_players:
        if all(action_indices[i] >= len(player_actions[i]) for i in range(num_players)):
            break
        
        current_player = game.current_player
        player_id = current_player
        
        if action_indices[player_id] >= len(player_actions[player_id]):
            game.current_player = (game.current_player + 1) % num_players
            continue
        
        action = player_actions[player_id][action_indices[player_id]]
        action_indices[player_id] += 1
        turn += 1
        
        current_room = game.player_locations[player_id]
        if current_room:
            if not is_movement_action(action):
                room_actions[(player_id, current_room.name)] = True
            
            if not player_paths[player_id] or player_paths[player_id][-1][0] != current_room.name:
                player_paths[player_id].append((current_room.name, turn))
        
        obs, reward, done, info = game.step(action, player_id)
        
        new_room = game.player_locations[player_id]
        if new_room and new_room != current_room:
            player_paths[player_id].append((new_room.name, turn))
        
        if done:
            break
    
    return player_paths, room_actions


def visualize_experiment(
    experiment_data: Dict,
    draw_paths: bool = True,
    figsize: Tuple[int, int] = (16, 12)
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    num_locations = experiment_data['num_locations']
    num_coins = experiment_data['num_coins']
    num_players = experiment_data['num_players']
    max_steps = experiment_data.get('max_steps')
    seed = experiment_data.get('seed')
    include_doors = experiment_data.get('include_doors', False)
    num_distractor_items = experiment_data.get('num_distractor_items', 0)
    coins_in_containers = experiment_data.get('coins_in_containers', False)
    limit_inventory_size = experiment_data.get('limit_inventory_size', True)
    connectivity = experiment_data.get('connectivity', 0.5)
    player_actions = experiment_data['player_actions']
    
    # Recreate the game state
    game = CoinCollectorGame(
        num_locations=num_locations,
        num_coins=num_coins,
        num_players=num_players,
        max_steps=max_steps,
        seed=seed,
        include_doors=include_doors,
        num_distractor_items=num_distractor_items,
        coins_in_containers=coins_in_containers,
        limit_inventory_size=limit_inventory_size,
        connectivity=connectivity
    )
    
    game.reset()
    
    player_paths, room_actions = simulate_player_paths(game, player_actions, num_players)
    
    game.reset()
    
    locations = game.locations
    num_locations = len(locations)
    base_font_size = 16
    base_icon_font_size = 14
    base_icon_radius = 0.08
    
    if num_locations <= 5:
        scale = 1.0
    elif num_locations <= 10:
        scale = 0.9
    elif num_locations <= 15:
        scale = 0.85
    elif num_locations <= 20:
        scale = 0.8
    else:
        scale = max(0.7, 0.8 - (num_locations - 20) * 0.01)
    
    base_font_size_calculated = int(base_font_size * scale)
    icon_font_size = int(base_icon_font_size * scale)
    icon_radius = base_icon_radius * scale
    
    coin_locations: set = set()
    for loc in locations:
        coins = [obj for obj in loc.collect_visible_objects() if 'coin' in obj.name]
        if coins:
            coin_locations.add(loc.name)
    
    player_start_location = locations[0].name
    
    nodes = {}
    edges = []
    for loc in locations:
        nodes[loc.name] = loc
        if loc.location_north:
            edges.append((loc.name, loc.location_north.name, 'north'))
        if loc.location_south:
            edges.append((loc.name, loc.location_south.name, 'south'))
        if loc.location_east:
            edges.append((loc.name, loc.location_east.name, 'east'))
        if loc.location_west:
            edges.append((loc.name, loc.location_west.name, 'west'))
    
    pos = _calculate_layout(nodes, edges, num_locations)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.axis('off')
    
    for source, target, direction in edges:
        x1, y1 = pos[source]
        x2, y2 = pos[target]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5, alpha=0.6, zorder=1)
    
    # Draw nodes (rooms) - matching visualize_map exactly
    def abbreviate_room_name(room_name: str, max_length: int = 10) -> str:
        abbreviations = {
            "living room": "living rm",
            "laundry room": "laundry rm",
            "bedroom": "bedrm",
            "bathroom": "bathrm",
            "kitchen": "kitchen",
            "pantry": "pantry",
            "corridor": "corridor",
            "backyard": "backyd",
            "driveway": "driveway",
            "street": "street",
            "supermarket": "supermkt"
        }
        
        parts = room_name.lower().split()
        if len(parts) >= 2 and parts[-1].isdigit():
            base_name = " ".join(parts[:-1])
            number = parts[-1]
            if base_name in abbreviations:
                return f"{abbreviations[base_name]} {number}"
            if len(base_name) > 6:
                return f"{base_name[:6].title()} {number}"
        
        room_lower = room_name.lower()
        if room_lower in abbreviations:
            return abbreviations[room_lower]
        
        if len(room_name) <= max_length:
            return room_name
        
        if " " in room_name:
            words = room_name.split()
            if len(words) == 2:
                return f"{words[0]} {words[1][:3].title()}"
            else:
                abbrev = words[0]
                for word in words[1:]:
                    abbrev += word[0].upper() if word else ""
                return abbrev
        
        return room_name[:max_length]
    
    room_box_dims = {}
    
    for name, loc in nodes.items():
        x, y = pos[name]
        
        facecolor = 'lightgrey'
        edgecolor = 'black'
        edgewidth = 2
        zorder = 2
        
        if name == player_start_location:
            facecolor = 'lightblue'
            edgecolor = 'blue'
            edgewidth = 4
            zorder = 3
        
        if name in coin_locations:
            if name == player_start_location:
                facecolor = 'lightcyan'
            else:
                facecolor = 'lightyellow'
            zorder = 3
        
        has_player = name == player_start_location
        has_coin = name in coin_locations
        num_icons = sum([has_player, has_coin])
        
        name_length = len(name)
        if num_locations <= 5:
            base_width = 1.0
        elif num_locations <= 10:
            base_width = 0.9
        elif num_locations <= 15:
            base_width = 0.8
        else:
            base_width = 0.75
        
        char_width_estimate = 0.045 * (base_font_size_calculated / base_font_size)
        width_needed = name_length * char_width_estimate
        icon_space_estimate = (icon_radius * 2 * num_icons) + (0.12 * base_width * (num_icons - 1)) if num_icons > 0 else 0
        width_needed += icon_space_estimate + 0.2
        
        box_width = max(base_width, width_needed)
        if num_locations <= 10:
            box_height = 0.35
        else:
            box_height = 0.3
        
        room_box_dims[name] = (box_width, box_height)
        
        box = mpatches.FancyBboxPatch(
            (x - box_width/2, y - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.03",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=max(1, edgewidth * scale),
            zorder=zorder
        )
        ax.add_patch(box)
        
        icon_offset_from_center = box_width / 2 - 0.12
        icon_x_start = x - icon_offset_from_center
        icon_spacing = 0.12 * box_width
        
        text_x = x
        if num_icons > 0:
            icon_space_needed = (icon_radius * 2 * num_icons) + (icon_spacing * (num_icons - 1))
            text_shift = min(icon_space_needed / 2 + 0.05, box_width / 2 - 0.1)
            text_x = x + text_shift
        
        display_name = abbreviate_room_name(name, max_length=12)
        
        name_length = len(display_name)
        if name_length > 12:
            font_size_adjusted = int(base_font_size_calculated * (1.0 - (name_length - 12) * 0.01))
            font_size_adjusted = max(int(base_font_size_calculated * 0.85), font_size_adjusted)
        else:
            font_size_adjusted = base_font_size_calculated
        
        ax.text(text_x, y, display_name, ha='center', va='center', 
                fontsize=font_size_adjusted, fontweight='bold', zorder=4)
        
        if has_player:
            person_circle = mpatches.Circle((icon_x_start, y), icon_radius, color='darkblue', zorder=5)
            ax.add_patch(person_circle)
            ax.text(icon_x_start, y, 'P', ha='center', va='center', 
                   fontsize=icon_font_size, color='white', fontweight='bold', zorder=6)
            icon_x_start += icon_spacing
        
        if has_coin:
            coin_circle = mpatches.Circle((icon_x_start, y), icon_radius, color='gold', zorder=5)
            ax.add_patch(coin_circle)
            ax.text(icon_x_start, y, 'C', ha='center', va='center', 
                   fontsize=icon_font_size, color='black', fontweight='bold', zorder=6)
    
    if draw_paths:
        colors = ['#FF0000', '#0066FF', '#00AA00', '#AA00AA', '#00AAAA', '#FFAA00']
        base_path_zorder = 10
        
        for player_id, path in enumerate(player_paths):
            if not path:
                continue
            
            color = colors[player_id % len(colors)]
            
            filtered_path = []
            for room_name, turn in path:
                if not filtered_path or filtered_path[-1][0] != room_name:
                    filtered_path.append((room_name, turn))
            
            marker_zorder = base_path_zorder + (len(player_paths) - player_id) + 1
            text_zorder = base_path_zorder + (len(player_paths) - player_id) + 2
            
            start_room, _ = filtered_path[0]
            if start_room in pos:
                x, y = pos[start_room]
                start_offset_radius = 0.12
                start_angle = ((num_players - 1 - player_id) * 2 * math.pi) / num_players if num_players > 1 else 0
                start_x = x + start_offset_radius * math.cos(start_angle)
                start_y = y + start_offset_radius * math.sin(start_angle)
                
                circle_radius = 0.075
                circle = mpatches.Circle((start_x, start_y), circle_radius, facecolor=color, 
                                       zorder=marker_zorder, alpha=0.9, linewidth=2, edgecolor='black')
                ax.add_patch(circle)
                ax.text(start_x, start_y, f'{player_id + 1}', ha='center', va='center',
                       fontsize=8, color='white', fontweight='bold', zorder=text_zorder)
            
            if len(filtered_path) < 2:
                continue
            
            offset_distance = 0.08
            
            for i in range(len(filtered_path) - 1):
                room1, turn1 = filtered_path[i]
                room2, turn2 = filtered_path[i + 1]
                
                if room1 in pos and room2 in pos and room1 in room_box_dims and room2 in room_box_dims:
                    x1_center, y1_center = pos[room1]
                    x2_center, y2_center = pos[room2]
                    
                    box_width1, box_height1 = room_box_dims[room1]
                    box_width2, box_height2 = room_box_dims[room2]
                    
                    dx = x2_center - x1_center
                    dy = y2_center - y1_center
                    length = (dx**2 + dy**2)**0.5
                    
                    if length > 0:
                        dir_x = dx / length
                        dir_y = dy / length
                        
                        half_width1 = box_width1 / 2
                        half_height1 = box_height1 / 2
                        
                        t_x_pos = (half_width1) / dir_x if dir_x > 0 else float('inf')
                        t_x_neg = (-half_width1) / dir_x if dir_x < 0 else float('inf')
                        t_y_pos = (half_height1) / dir_y if dir_y > 0 else float('inf')
                        t_y_neg = (-half_height1) / dir_y if dir_y < 0 else float('inf')
                        
                        t = min([t for t in [t_x_pos, t_x_neg, t_y_pos, t_y_neg] if t > 0])
                        
                        x1_edge = x1_center + dir_x * t
                        y1_edge = y1_center + dir_y * t
                        
                        half_width2 = box_width2 / 2
                        half_height2 = box_height2 / 2
                        
                        t_x_pos2 = (half_width2) / (-dir_x) if dir_x < 0 else float('inf')
                        t_x_neg2 = (-half_width2) / (-dir_x) if dir_x > 0 else float('inf')
                        t_y_pos2 = (half_height2) / (-dir_y) if dir_y < 0 else float('inf')
                        t_y_neg2 = (-half_height2) / (-dir_y) if dir_y > 0 else float('inf')
                        
                        t2 = min([t for t in [t_x_pos2, t_x_neg2, t_y_pos2, t_y_neg2] if t > 0])
                        
                        x2_edge = x2_center - dir_x * t2
                        y2_edge = y2_center - dir_y * t2
                        
                        perp_x = -dir_y
                        perp_y = dir_x
                        offset_x = perp_x * offset_distance * (player_id + 1) * 0.5
                        offset_y = perp_y * offset_distance * (player_id + 1) * 0.5
                        x1_offset = x1_edge + offset_x
                        y1_offset = y1_edge + offset_y
                        x2_offset = x2_edge + offset_x
                        y2_offset = y2_edge + offset_y
                    else:
                        x1_offset, y1_offset = x1_center, y1_center
                        x2_offset, y2_offset = x2_center, y2_center
                    
                    arrow_zorder = base_path_zorder + (len(player_paths) - player_id)
                    arrow = FancyArrowPatch((x1_offset, y1_offset), (x2_offset, y2_offset),
                                          arrowstyle='->', color=color, linewidth=4.0,
                                          alpha=0.9, zorder=arrow_zorder,
                                          mutation_scale=25, shrinkA=5, shrinkB=5)
                    ax.add_patch(arrow)
            
            start_room, _ = filtered_path[0]
            end_room, _ = filtered_path[-1]
            if end_room in pos and end_room != start_room:
                x, y = pos[end_room]
                square_size = 0.15
                offset_radius = 0.12
                
                angle = ((num_players - 1 - player_id) * 2 * math.pi) / num_players if num_players > 1 else 0
                square_x = x + offset_radius * math.cos(angle)
                square_y = y + offset_radius * math.sin(angle)
                
                square = mpatches.Rectangle((square_x - square_size/2, square_y - square_size/2), 
                                           square_size, square_size,
                                           facecolor=color, zorder=marker_zorder, alpha=0.9, 
                                           linewidth=2, edgecolor='black')
                ax.add_patch(square)
                ax.text(square_x, square_y, f'{player_id + 1}', ha='center', va='center',
                       fontsize=8, color='white', fontweight='bold', zorder=text_zorder)
            
            for room_name, _ in filtered_path:
                if (player_id, room_name) in room_actions and room_name in pos:
                    x, y = pos[room_name]
                    if room_name in room_box_dims:
                        box_width, box_height = room_box_dims[room_name]
                        action_x = x + box_width / 2 - 0.05
                        action_y = y + box_height / 2 - 0.05
                    else:
                        action_x = x + 0.3
                        action_y = y + 0.1
                    
                    star_size = 0.06
                    inner_radius = star_size * 0.38
                    
                    star_points = []
                    for i in range(10):
                        angle = (i * 2 * math.pi / 10) + math.pi / 2
                        if i % 2 == 0:
                            r = star_size
                        else:
                            r = inner_radius
                        star_points.append((action_x + r * math.cos(angle), 
                                          action_y + r * math.sin(angle)))
                    
                    codes = [MPLPath.MOVETO] + [MPLPath.LINETO] * 9 + [MPLPath.CLOSEPOLY]
                    star_points.append((0, 0))
                    star_path = MPLPath(star_points, codes)
                    star = PathPatch(star_path, facecolor=color, zorder=marker_zorder,
                                    alpha=0.9, linewidth=1.5, edgecolor='black')
                    ax.add_patch(star)
    
    all_x = [x for x, y in pos.values()]
    all_y = [y for x, y in pos.values()]
    ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
    
    if draw_paths:
        legend_elements = []
        for player_id in range(num_players):
            color = colors[player_id % len(colors)]
            legend_elements.append(
                mpatches.Patch(color=color, label=f'Player {player_id + 1}')
            )
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        model_name = experiment_data.get('model_name', 'Unknown')
        title = f'Experiment Visualization - {model_name}'
        if experiment_data.get('game_won'):
            title += ' (Won)'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=5)
    else:
        ax.set_title('Coin Collector Map', fontsize=16, fontweight='bold', pad=5)
    
    plt.tight_layout()
    
    return fig, ax, pos, room_box_dims


def main():
    parser = argparse.ArgumentParser(
        description='Visualize a single experiment run with player paths'
    )
    parser.add_argument(
        'experiment_dir',
        type=str,
        help='Path to experiment directory'
    )
    
    args = parser.parse_args()
    
    try:
        experiment_data, experiment_path = load_experiment(args.experiment_dir)
        
        fig_base, ax_base, pos, room_box_dims = visualize_experiment(experiment_data, draw_paths=False)
        base_map_path = experiment_path / "map_base.png"
        fig_base.savefig(base_map_path, dpi=150, bbox_inches='tight')
        print(f"Base map saved to {base_map_path}")
        plt.close(fig_base)
        
        fig_full, ax_full, pos, room_box_dims = visualize_experiment(experiment_data, draw_paths=True)
        full_map_path = experiment_path / "map_with_paths.png"
        fig_full.savefig(full_map_path, dpi=150, bbox_inches='tight')
        print(f"Full visualization saved to {full_map_path}")
        plt.close(fig_full)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
