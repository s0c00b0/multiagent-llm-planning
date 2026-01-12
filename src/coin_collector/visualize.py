"""Map visualization for Coin Collector game."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Optional, Tuple, Dict, Set
import math
import os
from pathlib import Path

from .objects import Room, FastObject
from .game import CoinCollectorGame


def visualize_map(
    game: CoinCollectorGame,
    output_path: Optional[str] = None,
    show_player_locations: bool = True,
    show_coin_locations: bool = True,
    figsize: Tuple[int, int] = (16, 12)
) -> None:
    if output_path is None:
        output_path = "out/map.png"
    elif not os.path.isabs(output_path):
        if not output_path.startswith("out/"):
            filename = os.path.basename(output_path)
            output_path = f"out/{filename}"
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    locations = game.locations
    if not locations:
        raise ValueError("Game has no locations. Call reset() first.")
    
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
    
    coin_locations: Set[str] = set()
    if show_coin_locations:
        for loc in locations:
            coins = [obj for obj in loc.collect_visible_objects() if 'coin' in obj.name]
            if coins:
                coin_locations.add(loc.name)
    
    player_start_location = locations[0].name if show_player_locations else None
    
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
        
        display_name = abbreviate_room_name(name, max_length=12)
        
        name_length = len(display_name)
        if name_length > 12:
            font_size_adjusted = int(base_font_size_calculated * (1.0 - (name_length - 12) * 0.01))
            font_size_adjusted = max(int(base_font_size_calculated * 0.85), font_size_adjusted)
        else:
            font_size_adjusted = base_font_size_calculated
        
        ax.text(text_x, y, display_name, ha='center', va='center', fontsize=font_size_adjusted, fontweight='bold', zorder=4)
        
        if has_player:
            person_circle = mpatches.Circle((icon_x_start, y), icon_radius, color='darkblue', zorder=5)
            ax.add_patch(person_circle)
            ax.text(icon_x_start, y, 'P', ha='center', va='center', fontsize=icon_font_size, color='white', fontweight='bold', zorder=6)
            icon_x_start += icon_spacing
        
        if has_coin:
            coin_circle = mpatches.Circle((icon_x_start, y), icon_radius, color='gold', zorder=5)
            ax.add_patch(coin_circle)
            ax.text(icon_x_start, y, 'C', ha='center', va='center', fontsize=icon_font_size, color='black', fontweight='bold', zorder=6)
    
    all_x = [x for x, y in pos.values()]
    all_y = [y for x, y in pos.values()]
    ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
    
    ax.set_title('Coin Collector', fontsize=16, fontweight='bold', pad=5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Map visualization saved to {output_path}")
    else:
        plt.show()


def _calculate_layout(nodes: Dict[str, Room], edges: List[Tuple[str, str, str]], num_locations: int) -> Dict[str, Tuple[float, float]]:
    pos = {}
    node_list = list(nodes.keys())
    if not node_list:
        return pos
    
    for node in node_list:
        pos[node] = None
    
    start_node = node_list[0]
    pos[start_node] = (0.0, 0.0)
    placed = {start_node}
    
    queue = [(start_node, 0.0, 0.0, 0)]
    used_positions = {(0.0, 0.0): start_node}
    
    while queue:
        current, x, y, depth = queue.pop(0)
        
        if current not in nodes:
            continue
        
        loc = nodes[current]
        if num_locations <= 5:
            spacing = 2.0
        elif num_locations <= 10:
            spacing = 1.6
        elif num_locations <= 15:
            spacing = 1.4
        elif num_locations <= 20:
            spacing = 1.2
        else:
            spacing = max(1.0, 1.2 - (num_locations - 20) * 0.01)
        
        if loc.location_north and loc.location_north.name not in placed:
            new_pos = (x, y + spacing)
            if new_pos in used_positions:
                new_pos = (x + 0.3, y + spacing)
            pos[loc.location_north.name] = new_pos
            placed.add(loc.location_north.name)
            used_positions[new_pos] = loc.location_north.name
            queue.append((loc.location_north.name, new_pos[0], new_pos[1], depth + 1))
        
        if loc.location_south and loc.location_south.name not in placed:
            new_pos = (x, y - spacing)
            if new_pos in used_positions:
                new_pos = (x + 0.3, y - spacing)
            pos[loc.location_south.name] = new_pos
            placed.add(loc.location_south.name)
            used_positions[new_pos] = loc.location_south.name
            queue.append((loc.location_south.name, new_pos[0], new_pos[1], depth + 1))
        
        if loc.location_east and loc.location_east.name not in placed:
            new_pos = (x + spacing, y)
            if new_pos in used_positions:
                new_pos = (x + spacing, y + 0.3)
            pos[loc.location_east.name] = new_pos
            placed.add(loc.location_east.name)
            used_positions[new_pos] = loc.location_east.name
            queue.append((loc.location_east.name, new_pos[0], new_pos[1], depth + 1))
        
        if loc.location_west and loc.location_west.name not in placed:
            new_pos = (x - spacing, y)
            if new_pos in used_positions:
                new_pos = (x - spacing, y + 0.3)
            pos[loc.location_west.name] = new_pos
            placed.add(loc.location_west.name)
            used_positions[new_pos] = loc.location_west.name
            queue.append((loc.location_west.name, new_pos[0], new_pos[1], depth + 1))
    
    for node in node_list:
        if pos[node] is None:
            offset = len(placed) * 2.0
            pos[node] = (offset, 0.0)
            placed.add(node)
    
    return pos

