"""Run Coin Collector game with LLM agents."""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    def load_dotenv(*args, **kwargs):
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.coin_collector import CoinCollectorGame


@dataclass
class GameStats:
    total_turns: int = 0
    player_turns: List[int] = None
    player_actions: List[List[str]] = None
    game_won: bool = False
    final_score: float = 0.0
    
    def __post_init__(self):
        if self.player_turns is None:
            self.player_turns = []
        if self.player_actions is None:
            self.player_actions = []


class LLMGameRunner:
    def __init__(self, model_type: str, api_key: Optional[str] = None, model_name: Optional[str] = None, api_params: Optional[Dict[str, Any]] = None):
        self.model_type = model_type.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.api_params = api_params or {}
        
        env_var_name = None
        if self.api_key is None:
            if self.model_type == 'openai':
                self.api_key = os.getenv('OPENAI_API_KEY')
                env_var_name = 'OPENAI_API_KEY'
            elif self.model_type == 'gemini':
                self.api_key = os.getenv('GEMINI_API_KEY')
                env_var_name = 'GEMINI_API_KEY'
        
        if self.api_key is None:
            raise ValueError(
                f"API key not found in environment variables for {model_type}.\n"
                f"Please set {env_var_name} in your .env file or as an environment variable.\n"
                f"Make sure python-dotenv is installed (pip install python-dotenv) and your .env file exists."
            )
        
        self._client_template = None
        self._init_client_template()
    
    def _init_client_template(self):
        if self.model_type == 'openai':
            try:
                import openai
                self._client_template = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")
        
        elif self.model_type == 'gemini':
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client_template = genai
            except ImportError:
                raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")
    
    def _create_player_clients(self, num_players: int) -> List[Any]:
        clients = []
        
        if self.model_type == 'openai':
            for _ in range(num_players):
                import openai
                clients.append({
                    'client': openai.OpenAI(api_key=self.api_key),
                    'messages': [],
                    'conversation_log': []
                })
        
        elif self.model_type == 'gemini':
            for _ in range(num_players):
                clients.append({
                    'model': self._client_template.GenerativeModel(self.model_name),
                    'conversation_log': []
                })
        
        return clients
    
    def _call_llm(self, prompt: str, player_id: int, player_clients: List[Any], system_prompt: Optional[str] = None) -> str:
        try:
            client_data = player_clients[player_id]
            
            if self.model_type == 'openai':
                client = client_data['client']
                messages = client_data['messages']
                conversation_log = client_data['conversation_log']
                
                if not messages and system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                    conversation_log.append({"role": "system", "content": system_prompt})
                
                messages.append({"role": "user", "content": prompt})
                conversation_log.append({"role": "user", "content": prompt})
                
                api_params = {
                    "model": self.model_name,
                    "messages": messages
                }
                
                if "max_completion_tokens" in self.api_params and self.api_params["max_completion_tokens"] is not None:
                    api_params["max_completion_tokens"] = self.api_params["max_completion_tokens"]
                elif "max_tokens" in self.api_params and self.api_params["max_tokens"] is not None:
                    api_params["max_tokens"] = self.api_params["max_tokens"]
                
                if "temperature" in self.api_params and self.api_params["temperature"] is not None:
                    api_params["temperature"] = self.api_params["temperature"]
                
                if "top_p" in self.api_params and self.api_params["top_p"] is not None:
                    api_params["top_p"] = self.api_params["top_p"]
                
                if "seed" in self.api_params and self.api_params["seed"] is not None:
                    seed_value = self.api_params["seed"]
                    if isinstance(seed_value, str):
                        try:
                            seed_value = int(seed_value)
                        except ValueError:
                            seed_value = None
                    if seed_value is not None:
                        api_params["seed"] = seed_value
                
                response = client.chat.completions.create(**api_params)
                
                content = response.choices[0].message.content
                if content is None:
                    print(f"Warning: OpenAI returned None content for Player {player_id + 1}")
                    return None
                
                assistant_message = content.strip()
                
                if not assistant_message:
                    print(f"Warning: OpenAI returned empty response for Player {player_id + 1}")
                    return None
                
                messages.append({"role": "assistant", "content": assistant_message})
                conversation_log.append({"role": "assistant", "content": assistant_message})
                
                return assistant_message
            
            elif self.model_type == 'gemini':
                conversation_log = client_data['conversation_log']
                model = client_data['model']
                
                full_prompt = prompt
                if system_prompt:
                    if not conversation_log:
                        conversation_log.append({"role": "system", "content": system_prompt})
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                
                conversation_log.append({"role": "user", "content": prompt})
                
                generation_config = {}
                if "temperature" in self.api_params and self.api_params["temperature"] is not None:
                    generation_config["temperature"] = self.api_params["temperature"]
                if "max_output_tokens" in self.api_params and self.api_params["max_output_tokens"] is not None:
                    generation_config["max_output_tokens"] = self.api_params["max_output_tokens"]
                elif "max_tokens" in self.api_params and self.api_params["max_tokens"] is not None:
                    generation_config["max_output_tokens"] = self.api_params["max_tokens"]
                if "top_p" in self.api_params and self.api_params["top_p"] is not None:
                    generation_config["top_p"] = self.api_params["top_p"]
                
                if generation_config:
                    response = model.generate_content(full_prompt, generation_config=generation_config)
                else:
                    response = model.generate_content(full_prompt)
                
                if response.text is None:
                    print(f"Warning: Gemini returned None content for Player {player_id + 1}")
                    return None
                
                text = response.text.strip()
                if not text:
                    print(f"Warning: Gemini returned empty response for Player {player_id + 1}")
                    return None
                
                conversation_log.append({"role": "assistant", "content": text})
                
                return text
        
        except Exception as e:
            print(f"Error calling LLM for Player {player_id + 1}: {e}")
            return None
    
    def _extract_action(self, llm_response: str, valid_actions: List[str]) -> Optional[str]:
        if llm_response is None or not llm_response.strip():
            return None
        
        response_lower = llm_response.lower().strip()
        
        for action in valid_actions:
            if action.lower() == response_lower:
                return action
        
        for action in valid_actions:
            if action.lower() in response_lower:
                return action
        
        action_keywords = {
            'move': ['move north', 'move south', 'move east', 'move west'],
            'take': [a for a in valid_actions if a.startswith('take ')],
            'open': [a for a in valid_actions if a.startswith('open ')],
            'close': [a for a in valid_actions if a.startswith('close ')],
            'inventory': ['inventory']
        }
        
        for keyword, actions in action_keywords.items():
            if keyword in response_lower:
                for action in actions:
                    if action in valid_actions:
                        return action
        
        return None
    
    def _create_prompt(self, observation: str, valid_actions: List[str], player_id: int, turn_num: int, num_coins: int) -> str:
        prompt = f"""You are Player {player_id + 1} in a Coin Collector game. Your goal is to collect all {num_coins} coin(s).

Current Situation (Turn {turn_num}):
{observation}

Available Actions:
"""
        for i, action in enumerate(valid_actions, 1):
            prompt += f"{i}. {action}\n"
        
        prompt += """
Choose one action from the list above. Respond with ONLY the action text, nothing else.
Example: "move north" or "take coin1" or "open door to north"
"""
        return prompt
    
    def run_game(
        self,
        num_locations: int = 20,
        num_coins: int = 3,
        num_players: int = 2,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
        include_doors: bool = True,
        num_distractor_items: int = 0,
        coins_in_containers: bool = False,
        limit_inventory_size: bool = True,
        connectivity: float = 0.5,
        verbose: bool = True,
        auto_save: bool = True,
        output_file: Optional[str] = None
    ) -> Tuple[GameStats, Dict[str, Any], Optional[Path]]:
        game = CoinCollectorGame(
            num_locations=num_locations,
            num_coins=num_coins,
            num_players=num_players,
            max_steps=max_steps,
            include_doors=include_doors,
            num_distractor_items=num_distractor_items,
            coins_in_containers=coins_in_containers,
            limit_inventory_size=limit_inventory_size,
            connectivity=connectivity,
            seed=seed
        )
        
        game.reset()
        
        stats = GameStats()
        stats.player_actions = [[] for _ in range(num_players)]
        
        if verbose:
            print(f"=== Starting Game ===")
            print(f"Model: {self.model_type} ({self.model_name})")
            print(f"Players: {num_players} (each with separate LLM instance)")
            print(f"Locations: {num_locations}")
            print(f"Coins: {num_coins}")
            print(f"Doors: {'enabled' if include_doors else 'disabled'}")
            print(f"Distractor items: {num_distractor_items}")
            print(f"Coins in containers: {'enabled' if coins_in_containers else 'disabled'}")
            print(f"Limit inventory size: {'enabled' if limit_inventory_size else 'disabled'}")
            if limit_inventory_size:
                print(f"  Inventory capacity: {num_coins + 1} items")
            print(f"Connectivity: {connectivity}")
            print(f"Max steps per player: {max_steps or 'unlimited'}")
            print()
        
        player_clients = self._create_player_clients(num_players)
        
        if verbose:
            print(f"Created {len(player_clients)} separate LLM instances (one per player)")
            print()
        
        max_steps_str = f"{max_steps} steps" if max_steps else "unlimited steps"
        doors_status = "enabled" if include_doors else "disabled"
        containers_status = "enabled" if coins_in_containers else "disabled"
        connectivity_desc = "minimal connections" if connectivity <= 0.3 else "maximum connections" if connectivity >= 0.7 else "moderate connections"
        
        system_prompt = f"""You are playing a text-based adventure game called Coin Collector.

GAME SETTINGS:
- Number of players: {num_players}
- Number of rooms/locations: {num_locations}
- Number of coins to collect: {num_coins}
- Maximum steps per player: {max_steps_str}
- Doors: {doors_status} (if enabled, you must open doors before moving through them)
- Distractor items: {num_distractor_items} (non-coin items in the game)
- Coins in containers: {containers_status} (if enabled, coins may be hidden inside containers like fridges, drawers, cabinets, etc.)
- Inventory limit: {'enabled (capacity: ' + str(num_coins + 1) + ' items)' if limit_inventory_size else 'disabled (unlimited)'}
- Connectivity: {connectivity} ({connectivity_desc})

GAME RULES:
- You are one of {num_players} players in this game.
- You can see other players when they are in the same room as you.
- You can see when players enter or exit your current room (including the direction they came from or went to).
- Your objective is to help collect all {num_coins} coin(s) in the game world as quickly as possible.
- You can move between rooms, open/close doors and containers, and take items.
- The game is won when ALL {num_coins} coin(s) are collected (by any player or combination of players).
- Each player has their own separate view of the game - you only see your own observations and actions.

RESPONSE FORMAT:
- Always respond with ONLY the action text from the available actions list.
- Do not include any explanation, reasoning, or additional text.
- Example valid responses: "move north", "take coin1", "open door to south", "inventory"
"""
        
        turn_num = 0
        max_turns = 200
        
        while turn_num < max_turns:
            current_player = game.current_player
            turn_num += 1
            
            observation = game.get_observation(current_player)
            valid_actions = [act[0] for act in game.last_valid_actions]
            
            if not valid_actions:
                if verbose:
                    print(f"Turn {turn_num}: No valid actions available. Game over.")
                break
            
            prompt = self._create_prompt(observation, valid_actions, current_player, turn_num, num_coins)
            
            if verbose:
                print(f"\n--- Turn {turn_num} - Player {current_player + 1} ---")
                print(f"Observation: {observation[:200]}..." if len(observation) > 200 else f"Observation: {observation}")
            
            llm_response = self._call_llm(prompt, current_player, player_clients, system_prompt)
            
            if verbose:
                if llm_response:
                    print(f"LLM Response: {llm_response}")
                else:
                    print(f"LLM Response: [BLANK/EMPTY]")
            
            action = self._extract_action(llm_response, valid_actions)
            
            if action is None:
                if verbose:
                    if llm_response:
                        print(f"Warning: Could not extract valid action from response '{llm_response}'. Skipping turn.")
                    else:
                        print(f"Warning: LLM returned blank/empty response. Skipping turn.")
                
                game.current_player = (current_player + 1) % num_players
                game._generate_valid_actions(game.current_player)
                continue
            
            if verbose:
                print(f"Selected Action: {action}")
            
            obs, reward, done, info = game.step(action)
            
            stats.total_turns = turn_num
            stats.player_turns.append(current_player)
            stats.player_actions[current_player].append(action)
            stats.final_score = info['scoreNormalized']
            
            if verbose:
                print(f"Reward: {reward}")
                print(f"Score: {info['scoreNormalized']:.2f} ({info['scoreRaw']}/{len(game.task_objects)})")
                print(f"Player steps: {info['playerSteps']}")
            
            if done:
                stats.game_won = info['taskSuccess']
                if verbose:
                    if stats.game_won:
                        print(f"\nGame Won! All players collected all coins together!")
                    else:
                        print(f"\nGame ended (step limit reached)")
                break
        
        if verbose:
            print(f"\n=== Game Summary ===")
            print(f"Total turns: {stats.total_turns}")
            print(f"Game won: {stats.game_won}")
            if stats.game_won:
                print(f"All players won together!")
            print(f"Final score: {stats.final_score:.2f}")
            print(f"Actions per player:")
            for i, actions in enumerate(stats.player_actions):
                print(f"  Player {i + 1}: {len(actions)} actions")
        
        game_info = {
            'total_turns': stats.total_turns,
            'game_won': stats.game_won,
            'final_score': stats.final_score,
            'player_actions': stats.player_actions,
            'player_steps': info['playerSteps'] if 'playerSteps' in locals() else [],
            'model_type': self.model_type,
            'model_name': self.model_name,
            'num_locations': num_locations,
            'num_coins': num_coins,
            'num_players': num_players,
            'max_steps': max_steps,
            'include_doors': include_doors,
            'num_distractor_items': num_distractor_items,
            'coins_in_containers': coins_in_containers,
            'limit_inventory_size': limit_inventory_size,
            'connectivity': connectivity,
            'seed': seed,
            'timestamp': datetime.now().isoformat()
        }
        
        run_dir = None
        if auto_save or output_file:
            output_base = Path('out')
            output_base.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            sanitized_model_name = self.model_name.replace('/', '_').replace('\\', '_').replace(' ', '_').replace(':', '_')
            sanitized_model_name = ''.join(c if c.isalnum() or c in ('_', '-', '.') else '_' for c in sanitized_model_name)
            folder_name = f"{sanitized_model_name}_{timestamp}"
            run_dir = output_base / folder_name
            run_dir.mkdir(exist_ok=True)
            
            if output_file:
                json_path = run_dir / Path(output_file).name
            else:
                json_filename = f"game_{self.model_type}_{timestamp}.json"
                json_path = run_dir / json_filename
            
            with open(json_path, 'w') as f:
                json.dump(game_info, f, indent=2)
            
            for player_id in range(num_players):
                player_log = player_clients[player_id]['conversation_log']
                player_filename = f"player_{player_id + 1}_conversation.txt"
                player_path = run_dir / player_filename
                
                with open(player_path, 'w') as f:
                    f.write(f"=== Player {player_id + 1} Complete Conversation Log ===\n\n")
                    for msg in player_log:
                        role = msg['role'].upper()
                        content = msg['content']
                        f.write(f"[{role}]\n{content}\n\n")
            
            if verbose:
                if output_file:
                    print(f"\nResults saved to directory: {run_dir}")
                    print(f"  - Game results: {Path(output_file).name}")
                else:
                    print(f"\nResults automatically saved to directory: {run_dir}")
                    print(f"  - Game results: {json_path.name}")
                for player_id in range(num_players):
                    print(f"  - Player {player_id + 1} log: player_{player_id + 1}_conversation.txt")
        
        return stats, game_info, run_dir


def load_env_file(env_path: str = '.env') -> None:
    if not DOTENV_AVAILABLE:
        print(f"Warning: python-dotenv not installed. Install with: pip install python-dotenv")
        print(f"Environment variables from {env_path} will not be loaded automatically.")
        print(f"Please set environment variables manually or install python-dotenv.")
        return
    
    env_file = Path(env_path)
    if env_file.exists():
        load_dotenv(env_path, override=True)
    elif env_path == '.env':
        print(f"Warning: .env file not found at {env_file.absolute()}")
        print(f"Please create a .env file with your API keys or set environment variables manually.")


def load_config(config_path: str) -> Dict[str, Any]:
    if not YAML_AVAILABLE:
        print(f"Error: PyYAML not installed. Install with: pip install pyyaml")
        print(f"Cannot read config file {config_path}. Exiting.")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config if config else {}
    except FileNotFoundError:
        print(f"Error: Config file {config_path} not found. Exiting.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Run Coin Collector game with LLM agents')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to YAML config file (default: config.yaml)')
    
    args = parser.parse_args()
    
    load_env_file('.env')
    config = load_config(args.config)
    
    model = config.get('model')
    model_name = config.get('model_name')
    api_params = config.get('api_params', {})
    num_locations = config.get('num_locations', 20)
    num_coins = config.get('num_coins', 3)
    num_players = config.get('num_players', 2)
    max_steps = config.get('max_steps')
    seed = config.get('seed')
    
    include_doors = config.get('include_doors', True)
    if isinstance(include_doors, str):
        include_doors_str = include_doors.lower()
        include_doors = include_doors_str in ('true', '1', 'yes', 'on')
    
    num_distractor_items = config.get('num_distractor_items', 0)
    
    coins_in_containers = config.get('coins_in_containers', False)
    if isinstance(coins_in_containers, str):
        coins_in_containers_str = coins_in_containers.lower()
        coins_in_containers = coins_in_containers_str in ('true', '1', 'yes', 'on')
    
    limit_inventory_size = config.get('limit_inventory_size', True)
    if isinstance(limit_inventory_size, str):
        limit_inventory_size_str = limit_inventory_size.lower()
        limit_inventory_size = limit_inventory_size_str in ('true', '1', 'yes', 'on')
    
    connectivity = config.get('connectivity', 0.5)
    if isinstance(connectivity, str):
        try:
            connectivity = float(connectivity)
        except ValueError:
            connectivity = 0.5
    
    quiet = config.get('quiet', False)
    output_file = config.get('output')
    no_auto_save = config.get('no_auto_save', False)
    
    
    try:
        runner = LLMGameRunner(
            model_type=model,
            api_key=None,
            model_name=model_name,
            api_params=api_params
        )
        
        stats, game_info, run_dir = runner.run_game(
            num_locations=num_locations,
            num_coins=num_coins,
            num_players=num_players,
            max_steps=max_steps,
            seed=seed,
            include_doors=include_doors,
            num_distractor_items=num_distractor_items,
            coins_in_containers=coins_in_containers,
            limit_inventory_size=limit_inventory_size,
            connectivity=connectivity,
            verbose=not quiet,
            auto_save=not no_auto_save,
            output_file=output_file
        )
        
        sys.exit(0 if stats.game_won else 1)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
