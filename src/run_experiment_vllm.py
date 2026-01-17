"""Run Coin Collector game with LLM agents using VLLM."""

import os
import sys
import json
import argparse
import re
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
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
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
    def __init__(self, model_type: str, model_name: Optional[str] = None, api_params: Optional[Dict[str, Any]] = None):
        self.model_type = model_type.lower()
        self.model_name = model_name or "Qwen/Qwen3-32B"
        self.api_params = api_params or {}
        
        self._llm = None
        self._init_llm()
    
    def _init_llm(self):
        if self.model_type == 'vllm':
            try:
                from vllm import LLM
                from vllm.sampling_params import SamplingParams
                
                max_tokens = 500
                if "max_tokens" in self.api_params and self.api_params["max_tokens"] is not None:
                    max_tokens = self.api_params["max_tokens"]
                elif "max_output_tokens" in self.api_params and self.api_params["max_output_tokens"] is not None:
                    max_tokens = self.api_params["max_output_tokens"]
                elif "max_completion_tokens" in self.api_params and self.api_params["max_completion_tokens"] is not None:
                    max_tokens = self.api_params["max_completion_tokens"]
                
                temperature = self.api_params.get("temperature", 1.0) if "temperature" in self.api_params and self.api_params["temperature"] is not None else 1.0
                top_p = self.api_params.get("top_p") if "top_p" in self.api_params and self.api_params["top_p"] is not None else None
                seed = self.api_params.get("seed") if "seed" in self.api_params and self.api_params["seed"] is not None else None
                
                if isinstance(seed, str):
                    try:
                        seed = int(seed)
                    except ValueError:
                        seed = None
                
                sampling_params_dict = {
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                if top_p is not None:
                    sampling_params_dict["top_p"] = top_p
                if seed is not None:
                    sampling_params_dict["seed"] = seed
                
                self._sampling_params = SamplingParams(**sampling_params_dict)
                
                llm_kwargs = {}
                
                if "gpu_memory_utilization" in self.api_params and self.api_params["gpu_memory_utilization"] is not None:
                    gpu_mem_util = self.api_params["gpu_memory_utilization"]
                    if isinstance(gpu_mem_util, str):
                        try:
                            gpu_mem_util = float(gpu_mem_util)
                        except ValueError:
                            gpu_mem_util = None
                    if gpu_mem_util is not None:
                        llm_kwargs["gpu_memory_utilization"] = gpu_mem_util
                
                if "max_model_len" in self.api_params and self.api_params["max_model_len"] is not None:
                    max_len = self.api_params["max_model_len"]
                    if isinstance(max_len, str):
                        try:
                            max_len = int(max_len)
                        except ValueError:
                            max_len = None
                    if max_len is not None:
                        llm_kwargs["max_model_len"] = max_len
                
                hf_home = os.getenv('HF_HOME')
                if not hf_home:
                    hf_home = os.path.expanduser('~/.cache/huggingface')
                    os.environ['HF_HOME'] = hf_home
                
                os.environ['TRANSFORMERS_OFFLINE'] = '0'
                os.environ['HF_HUB_OFFLINE'] = '0'
                
                try:
                    from huggingface_hub import snapshot_download
                    cache_dir = os.path.join(hf_home, 'hub')
                    if '/' in self.model_name:
                        org, model = self.model_name.split('/', 1)
                        model_cache_path = os.path.join(cache_dir, f'models--{org}--{model.replace("/", "--")}')
                    else:
                        model_cache_path = os.path.join(cache_dir, f'models--{self.model_name.replace("/", "--")}')
                except ImportError:
                    pass
                
                print(f"Initializing VLLM model: {self.model_name}")
                self._llm = LLM(model=self.model_name, **llm_kwargs)
                
            except ImportError:
                raise ImportError("vllm package not installed. Install with: pip install vllm")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Use 'vllm' for VLLM models.")
    
    def _create_player_clients(self, num_players: int) -> List[Any]:
        clients = []
        for _ in range(num_players):
            clients.append({
                'conversation_log': []
            })
        return clients
    
    def _call_llm(self, prompt: str, player_id: int, player_clients: List[Any], system_prompt: Optional[str] = None) -> str:
        try:
            client_data = player_clients[player_id]
            conversation_log = client_data['conversation_log']
            
            messages = []
            
            if system_prompt:
                if not conversation_log or not any(msg.get('role') == 'system' for msg in conversation_log):
                    messages.append({"role": "system", "content": system_prompt})
                    conversation_log.append({"role": "system", "content": system_prompt})
            
            for msg in conversation_log:
                if msg.get('role') != 'system':
                    messages.append({"role": msg['role'], "content": msg['content']})
            
            messages.append({"role": "user", "content": prompt})
            conversation_log.append({"role": "user", "content": prompt})
            
            formatted_prompt = self._format_messages_for_vllm(messages)
            outputs = self._llm.generate([formatted_prompt], self._sampling_params)
            
            if not outputs or not outputs[0].outputs:
                print(f"Warning: VLLM returned empty output for Player {player_id + 1}")
                return None
            
            assistant_message = outputs[0].outputs[0].text.strip()
            
            if not assistant_message:
                print(f"Warning: VLLM returned empty response for Player {player_id + 1}")
                return None
            
            conversation_log.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
        
        except Exception as e:
            print(f"Error calling LLM for Player {player_id + 1}: {e}")
            return None
    
    def _format_messages_for_vllm(self, messages: List[Dict[str, str]]) -> str:
        try:
            tokenizer = None
            if hasattr(self._llm, 'llm_engine') and hasattr(self._llm.llm_engine, 'tokenizer'):
                if hasattr(self._llm.llm_engine.tokenizer, 'tokenizer'):
                    tokenizer = self._llm.llm_engine.tokenizer.tokenizer
                elif hasattr(self._llm.llm_engine.tokenizer, 'apply_chat_template'):
                    tokenizer = self._llm.llm_engine.tokenizer
            
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg.get('role', 'user'),
                    "content": msg.get('content', '')
                })
            
            if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
                try:
                    prompt = tokenizer.apply_chat_template(
                        formatted_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    return prompt
                except Exception:
                    pass
        except Exception:
            pass
        
        formatted_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                formatted_parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
            elif role == 'user':
                formatted_parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
            elif role == 'assistant':
                formatted_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
        
        formatted_parts.append("<|im_start|>assistant\n")
        return "".join(formatted_parts)
    
    def _strip_reasoning_tokens(self, text: str) -> str:
        if not text:
            return text
        
        pattern = r'<think>.*?</think>'
        cleaned = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        pattern_alt = r'&lt;think&gt;.*?&lt;/think&gt;'
        cleaned = re.sub(pattern_alt, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'\n\s*\n+', '\n', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _extract_action(self, llm_response: str, valid_actions: List[str]) -> Optional[str]:
        if llm_response is None or not llm_response.strip():
            return None
        
        cleaned_response = self._strip_reasoning_tokens(llm_response)
        
        if not cleaned_response:
            return None
        
        response_lower = cleaned_response.lower().strip()
        
        for action in valid_actions:
            if action.lower() == response_lower:
                return action
        
        lines = [line.strip() for line in cleaned_response.split('\n')]
        for line in lines:
            line_lower = line.lower().strip()
            for action in valid_actions:
                if action.lower() == line_lower:
                    return action
        
        sorted_actions = sorted(valid_actions, key=lambda x: len(x), reverse=True)
        for action in sorted_actions:
            action_lower = action.lower()
            if re.search(r'\b' + re.escape(action_lower) + r'\b', response_lower):
                return action
        
        for action in sorted_actions:
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
            print(f"Players: {num_players} (sharing VLLM instance)")
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
            print(f"Created {len(player_clients)} player conversation logs (sharing VLLM instance)")
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
    parser = argparse.ArgumentParser(description='Run Coin Collector game with LLM agents using VLLM')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to YAML config file (default: config.yaml)')
    
    args = parser.parse_args()
    
    load_env_file('.env')
    config = load_config(args.config)
    
    model = config.get('model', 'vllm')
    model_name = config.get('model_name', 'Qwen/Qwen3-32B')
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
