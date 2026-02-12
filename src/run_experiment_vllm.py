"""Run Coin Collector game with LLM agents using VLLM."""

import os
import sys
import json
import argparse
import re
import subprocess
import tempfile
import requests
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
    
    def _load_prompt_file(self, filename: str) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        prompt_file = os.path.join(project_root, "prompts", filename)
        with open(prompt_file, 'r') as f:
            return f.read()

    def _load_prompt_template(self, prompt_version: str = "standard") -> str:
        return self._load_prompt_file(f"{prompt_version}.txt")
    
    def _load_pddl_system_template(self) -> str:
        return self._load_prompt_file("pddl_system.txt")

    def _load_pddl_user_template(self) -> str:
        return self._load_prompt_file("pddl_user.txt")

    def _load_action_user_template(self) -> str:
        return self._load_prompt_file("action_user.txt")

    def _load_pddl_predict_other_user_template(self) -> str:
        return self._load_prompt_file("pddl_predict_other_user.txt")

    def _load_pddl_own_plan_user_template(self) -> str:
        return self._load_prompt_file("pddl_own_plan_user.txt")

    def _load_pddl_two_player_system_template(self) -> str:
        return self._load_prompt_file("pddl_two_player_system.txt")
    
    def _create_player_clients(self, num_players: int) -> List[Any]:
        clients = []
        for _ in range(num_players):
            clients.append({
                'conversation_log': []
            })
        return clients

    def _append_to_conversation_file(self, run_dir: Optional[Path], player_id: int, role: str, content: str) -> None:
        if run_dir is None:
            return
        player_path = run_dir / f"player_{player_id + 1}_conversation.txt"
        write_header = not player_path.exists()
        with open(player_path, 'a') as f:
            if write_header:
                f.write(f"=== Player {player_id + 1} Complete Conversation Log ===\n\n")
            f.write(f"[{role.upper()}]\n{content}\n\n")

    def _log_message(self, player_clients: List[Any], player_id: int, role: str, content: str, run_dir: Optional[Path] = None) -> None:
        player_clients[player_id]['conversation_log'].append({"role": role, "content": content})
        if run_dir:
            self._append_to_conversation_file(run_dir, player_id, role, content)
    
    def _call_llm(self, prompt: str, player_id: int, player_clients: List[Any], system_prompt: Optional[str] = None, run_dir: Optional[Path] = None) -> str:
        try:
            client_data = player_clients[player_id]
            conversation_log = client_data['conversation_log']
            
            messages = []
            
            if system_prompt:
                if not conversation_log or not any(msg.get('role') == 'system' for msg in conversation_log):
                    messages.append({"role": "system", "content": system_prompt})
                    self._log_message(player_clients, player_id, "system", system_prompt, run_dir)
            
            for msg in conversation_log:
                if msg.get('role') != 'system':
                    messages.append({"role": msg['role'], "content": msg['content']})
            
            messages.append({"role": "user", "content": prompt})
            self._log_message(player_clients, player_id, "user", prompt, run_dir)
            
            formatted_prompt = self._format_messages_for_vllm(messages)
            outputs = self._llm.generate([formatted_prompt], self._sampling_params)
            
            if not outputs or not outputs[0].outputs:
                print(f"Warning: VLLM returned empty output for Player {player_id + 1}")
                return None
            
            assistant_message = outputs[0].outputs[0].text.strip()
            
            if not assistant_message:
                print(f"Warning: VLLM returned empty response for Player {player_id + 1}")
                return None
            
            self._log_message(player_clients, player_id, "assistant", assistant_message, run_dir)
            
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
    
    def _create_prompt(self, observation: str, valid_actions: List[str], player_id: int, turn_num: int, num_coins: int, use_pddl: bool = False, feedback: Optional[str] = None) -> str:
        if use_pddl:
            valid_actions_list = ""
            for i, action in enumerate(valid_actions, 1):
                valid_actions_list += f"{i}. {action}\n"
            
            feedback_section = ""
            if feedback:
                feedback_section = f"\nFEEDBACK FROM PREVIOUS ATTEMPT:\n{feedback}\n"
            
            user_template = self._load_pddl_user_template()
            prompt = self._load_pddl_user_template().format(
                turn_num=turn_num,
                observation=observation,
                valid_actions_list=valid_actions_list,
                feedback_section=feedback_section
            )
        else:
            valid_actions_list = ""
            for i, action in enumerate(valid_actions, 1):
                valid_actions_list += f"{i}. {action}\n"
            prompt = self._load_action_user_template().format(
                player_id=player_id + 1,
                num_coins=num_coins,
                turn_num=turn_num,
                observation=observation,
                valid_actions_list=valid_actions_list
            )
        return prompt
    
    def _extract_pddl_files(self, llm_response: str) -> Tuple[Optional[str], Optional[str]]:
        code_block_pattern = r'```(?:pddl)?\s*\n?(.*?)```'
        code_blocks = re.findall(code_block_pattern, llm_response, re.DOTALL | re.IGNORECASE)
        
        domain = None
        problem = None
        
        for block in code_blocks:
            block = block.strip()
            if '(define (domain' in block or '(define(domain' in block:
                if domain is None:
                    domain = block
            if '(define (problem' in block or '(define(problem' in block:
                if problem is None:
                    problem = block
        
        if domain is None:
            domain_start = llm_response.find('(define (domain')
            if domain_start == -1:
                domain_start = llm_response.find('(define(domain')
            
            if domain_start != -1:
                paren_count = 0
                domain_end = domain_start
                for i in range(domain_start, len(llm_response)):
                    char = llm_response[i]
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                        if paren_count == 0:
                            domain_end = i + 1
                            domain = llm_response[domain_start:domain_end]
                            break
        
        if problem is None:
            problem_start = llm_response.find('(define (problem')
            if problem_start == -1:
                problem_start = llm_response.find('(define(problem')
            
            if problem_start != -1:
                paren_count = 0
                problem_end = problem_start
                for i in range(problem_start, len(llm_response)):
                    char = llm_response[i]
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                        if paren_count == 0:
                            problem_end = i + 1
                            problem = llm_response[problem_start:problem_end]
                            break
        
        if domain:
            domain = re.sub(r'^```pddl\s*', '', domain, flags=re.MULTILINE | re.IGNORECASE)
            domain = re.sub(r'^```\s*', '', domain, flags=re.MULTILINE)
            domain = re.sub(r'```\s*$', '', domain, flags=re.MULTILINE)
            domain = domain.strip()
            domain = '\n'.join(line.rstrip() for line in domain.split('\n'))
        
        if problem:
            problem = re.sub(r'^```pddl\s*', '', problem, flags=re.MULTILINE | re.IGNORECASE)
            problem = re.sub(r'^```\s*', '', problem, flags=re.MULTILINE)
            problem = re.sub(r'```\s*$', '', problem, flags=re.MULTILINE)
            problem = problem.strip()
            problem = '\n'.join(line.rstrip() for line in problem.split('\n'))
        
        return domain, problem
    
    def _validate_pddl_syntax(self, domain: str, problem: str, verbose: bool = False, save_dir: Optional[Path] = None, turn_num: Optional[int] = None, attempt: Optional[int] = None    ) -> Tuple[bool, Optional[str]]:
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"_turn{turn_num}_attempt{attempt}" if turn_num is not None and attempt is not None else ""
            domain_save_path = save_dir / f"domain{suffix}.pddl"
            problem_save_path = save_dir / f"problem{suffix}.pddl"
            with open(domain_save_path, 'w') as f:
                f.write(domain)
            with open(problem_save_path, 'w') as f:
                f.write(problem)
            if verbose:
                print(f"  PDDL files saved for debugging:")
                print(f"    Domain: {domain_save_path}")
                print(f"    Problem: {problem_save_path}")
        
        return True, None
    
    def _call_pddl_solver(self, domain: str, problem: str, verbose: bool = False) -> Tuple[Optional[List[str]], Optional[str]]:
        import tempfile
        import os
        import subprocess
        
        project_root = os.path.dirname(os.path.dirname(__file__))
        local_bin = os.path.join(project_root, 'venv', 'local', 'bin')
        local_etc = os.path.join(project_root, 'venv', 'local', 'etc', 'apptainer')
        env = os.environ.copy()
        if os.path.exists(local_bin):
            env['PATH'] = f"{local_bin}:{env.get('PATH', '')}"
        if os.path.exists(local_etc):
            env['APPTAINER_CONFDIR'] = local_etc
        
        temp_dir = tempfile.mkdtemp()
        try:
            domain_path = os.path.join(temp_dir, 'domain.pddl')
            problem_path = os.path.join(temp_dir, 'problem.pddl')
            plan_path = os.path.join(temp_dir, 'plan')
            
            with open(domain_path, 'w') as f:
                f.write(domain)
            with open(problem_path, 'w') as f:
                f.write(problem)
            
            try:
                if verbose:
                    print(f"  Calling dual-bfws-ffparser via planutils...")
                
                result = subprocess.run(
                    ['planutils', 'run', 'dual-bfws-ffparser', domain_path, problem_path],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=env,
                    cwd=temp_dir
                )
                
                if verbose:
                    if result.stdout:
                        print(f"  Solver stdout (first 500 chars): {result.stdout[:500]}")
                    if result.stderr:
                        print(f"  Solver stderr (first 500 chars): {result.stderr[:500]}")
                    print(f"  Solver return code: {result.returncode}")
                
                if result.returncode == 0:
                    plan_lines = []
                    
                    if os.path.exists(plan_path):
                        with open(plan_path, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if not line or line.startswith(';'):
                                    continue
                                if line.startswith('('):
                                    action_str = line.strip('()').strip()
                                    if action_str:
                                        plan_lines.append(action_str)
                                else:
                                    plan_lines.append(line)
                    
                    if plan_lines:
                        if verbose:
                            print(f"  Parsed {len(plan_lines)} actions from plan")
                        return plan_lines, None
                    else:
                        error_msg = "Solver returned success but no plan found"
                        if verbose:
                            print(f"  {error_msg}")
                            print(f"  Full stdout: {result.stdout}")
                        return None, error_msg
                else:
                    error_msg = result.stderr or result.stdout or f"Return code: {result.returncode}"
                    if verbose:
                        print(f"  dual-bfws-ffparser failed with return code {result.returncode}")
                        if result.stderr:
                            print(f"  Full stderr: {result.stderr}")
                        if result.stdout:
                            print(f"  Full stdout: {result.stdout}")
                    return None, error_msg
            finally:
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except:
                    pass
        except FileNotFoundError:
            error_msg = "planutils command not found. Please install planutils: pip install planutils"
            if verbose:
                print(f"  Error: {error_msg}")
            return None, error_msg
        except subprocess.TimeoutExpired:
            error_msg = "dual-bfws-ffparser solver timed out after 120 seconds"
            if verbose:
                print(f"  Error: {error_msg}")
            return None, error_msg
        except Exception as e:
            error_msg = f"Error calling dual-bfws-ffparser: {str(e)}"
            if verbose:
                print(f"  Exception: {error_msg}")
                import traceback
                print(f"  Traceback: {traceback.format_exc()[:500]}")
            return None, error_msg
    
    def _parse_plan_action(self, plan_action: str, valid_actions: List[str]) -> Optional[str]:
        action_clean = plan_action.strip().strip('()').strip()
        
        if action_clean in valid_actions:
            return action_clean
        
        action_parts = action_clean.split()
        if not action_parts:
            return None
        
        action_name = action_parts[0].lower()
        
        if action_name == 'move':
            if len(action_parts) >= 2:
                direction = action_parts[-1].lower()
                valid_directions = ['north', 'south', 'east', 'west']
                if direction in valid_directions:
                    move_action = f"move {direction}"
                    if move_action in valid_actions:
                        return move_action
                    for valid_action in valid_actions:
                        if valid_action.lower() == move_action.lower():
                            return valid_action
        elif action_name == 'take' or action_name == 'pickup':
            if len(action_parts) > 1:
                obj = action_parts[1].lower()
                take_action = f"take {obj}"
                if take_action in valid_actions:
                    return take_action
        elif action_name == 'open' or action_name == 'open_door':
            if 'door' in action_clean.lower():
                if 'north' in action_clean.lower():
                    return "open door to north" if "open door to north" in valid_actions else None
                elif 'south' in action_clean.lower():
                    return "open door to south" if "open door to south" in valid_actions else None
                elif 'east' in action_clean.lower():
                    return "open door to east" if "open door to east" in valid_actions else None
                elif 'west' in action_clean.lower():
                    return "open door to west" if "open door to west" in valid_actions else None
        elif action_name == 'close' or action_name == 'close_door':
            if 'door' in action_clean.lower():
                if 'north' in action_clean.lower():
                    return "close door to north" if "close door to north" in valid_actions else None
                elif 'south' in action_clean.lower():
                    return "close door to south" if "close door to south" in valid_actions else None
                elif 'east' in action_clean.lower():
                    return "close door to east" if "close door to east" in valid_actions else None
                elif 'west' in action_clean.lower():
                    return "close door to west" if "close door to west" in valid_actions else None
        
        action_lower = action_clean.lower()
        for valid_action in valid_actions:
            if valid_action.lower() in action_lower or action_lower in valid_action.lower():
                return valid_action
        
        return None
    
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
        prompt_version: str = "standard",
        use_pddl: bool = False,
        verbose: bool = True,
        auto_save: bool = True,
        output_file: Optional[str] = None,
        run_dir_suffix: Optional[str] = None
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
        
        run_dir = None
        if auto_save or output_file:
            output_base = Path('out')
            output_base.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            sanitized_model_name = self.model_name.replace('/', '_').replace('\\', '_').replace(' ', '_').replace(':', '_')
            sanitized_model_name = ''.join(c if c.isalnum() or c in ('_', '-', '.') else '_' for c in sanitized_model_name)
            folder_name = f"{sanitized_model_name}_{timestamp}"
            if run_dir_suffix:
                folder_name = f"{folder_name}_{run_dir_suffix}"
            run_dir = output_base / folder_name
            run_dir.mkdir(exist_ok=True)
            
            pddl_debug_dir = run_dir / 'pddl_debug'
            pddl_debug_dir.mkdir(exist_ok=True)
        
        max_steps_str = f"{max_steps} steps" if max_steps else "unlimited steps"
        doors_status = "enabled" if include_doors else "disabled"
        containers_status = "enabled" if coins_in_containers else "disabled"
        connectivity_desc = "minimal connections" if connectivity <= 0.3 else "maximum connections" if connectivity >= 0.7 else "moderate connections"
        inventory_limit_str = f"enabled (capacity: {num_coins + 1} items)" if limit_inventory_size else "disabled (unlimited)"
        
        containers_efficiency_note = ""
        if prompt_version == "improved" and not coins_in_containers:
            containers_efficiency_note = "Only check containers if coins can actually be inside them - don't waste actions checking containers when coins are only found in plain sight. "
        
        prompt_template = self._load_prompt_template(prompt_version)
        
        system_prompt = prompt_template.format(
            num_players=num_players,
            num_locations=num_locations,
            num_coins=num_coins,
            max_steps_str=max_steps_str,
            doors_status=doors_status,
            num_distractor_items=num_distractor_items,
            containers_status=containers_status,
            inventory_limit_str=inventory_limit_str,
            connectivity=connectivity,
            connectivity_desc=connectivity_desc,
            containers_efficiency_note=containers_efficiency_note
        )
        
        if use_pddl and 'Qwen3-32B' in self.model_name:
            system_prompt = system_prompt + "\n\n" + self._load_pddl_system_template().format(num_coins=num_coins)
        
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
            
            use_pddl_mode = use_pddl and 'Qwen3-32B' in self.model_name
            
            if use_pddl_mode:
                action = None
                feedback = None
                max_pddl_retries = 3
                executed_any = False
                
                if num_players == 2:
                    other_player = (current_player + 1) % num_players
                    two_player_system = system_prompt + "\n\n" + self._load_pddl_two_player_system_template().format(
                        current_player_id=current_player + 1,
                        other_player_id=other_player + 1,
                        num_coins=num_coins
                    )
                    game._generate_valid_actions(other_player)
                    other_observation = game.get_observation(other_player)
                    other_valid_actions = [act[0] for act in game.last_valid_actions]
                    game._generate_valid_actions(current_player)
                    other_valid_actions_list = "\n".join(f"{i+1}. {a}" for i, a in enumerate(other_valid_actions))
                    
                    for retry in range(max_pddl_retries):
                        if verbose and retry == 0:
                            print(f"\n--- Turn {turn_num} - Player {current_player + 1} (Two-Player PDDL) ---")
                            print(f"Step 1: Predicting other player's PDDL...")
                        predict_other_user = self._load_pddl_predict_other_user_template().format(
                            other_player_id=other_player + 1,
                            turn_num=turn_num,
                            other_observation=other_observation,
                            other_valid_actions_list=other_valid_actions_list
                        )
                        llm_response_other = self._call_llm(predict_other_user, current_player, player_clients, two_player_system, run_dir)
                        llm_response_other_cleaned = self._strip_reasoning_tokens(llm_response_other) if llm_response_other else ""
                        domain_other, problem_other = self._extract_pddl_files(llm_response_other_cleaned)
                        if not domain_other or not problem_other:
                            feedback = "ERROR: Could not extract PDDL from your prediction of the other player. Please provide both domain and problem in ```pddl blocks."
                            if verbose:
                                print(f"  Step 1 failed: Could not extract other player's PDDL (attempt {retry + 1})")
                            if retry < max_pddl_retries - 1:
                                continue
                            break
                        plan_other, solver_error_other = self._call_pddl_solver(domain_other, problem_other, verbose=verbose)
                        plan_other_formatted = "\n".join(f"{i+1}. {a}" for i, a in enumerate(plan_other)) if plan_other else "(solver returned no plan)"
                        solver_result_other = f"PDDL Solver Result (other player's predicted plan):\n{plan_other_formatted}" if plan_other else f"PDDL Solver Error (other player's prediction):\n{solver_error_other or 'No plan found'}"
                        self._log_message(player_clients, current_player, "user", solver_result_other, run_dir)
                        if verbose:
                            print(f"  Step 2: Other player's predicted plan (solved): {plan_other_formatted[:200]}..." if len(plan_other_formatted) > 200 else f"  Step 2: Other player's predicted plan: {plan_other_formatted}")
                        valid_actions_list = "\n".join(f"{i+1}. {a}" for i, a in enumerate(valid_actions))
                        feedback_section = f"\nFEEDBACK FROM PREVIOUS ATTEMPT:\n{feedback}\n" if feedback else ""
                        own_plan_user = self._load_pddl_own_plan_user_template().format(
                            other_player_predicted_plan=plan_other_formatted,
                            current_player_id=current_player + 1,
                            turn_num=turn_num,
                            observation=observation,
                            valid_actions_list=valid_actions_list,
                            feedback_section=feedback_section
                        )
                        if verbose:
                            print(f"  Step 3: Generating your own PDDL...")
                        llm_response = self._call_llm(own_plan_user, current_player, player_clients, two_player_system, run_dir)
                        llm_response_cleaned = self._strip_reasoning_tokens(llm_response) if llm_response else ""
                        domain, problem = self._extract_pddl_files(llm_response_cleaned)
                        if not domain or not problem:
                            feedback = "ERROR: Could not extract PDDL from your own plan. Please provide both domain and problem in ```pddl blocks."
                            if verbose:
                                print(f"  Step 3 failed: Could not extract your PDDL (attempt {retry + 1})")
                            if retry < max_pddl_retries - 1:
                                continue
                            break
                        if verbose:
                            print(f"  Step 4: Solving your PDDL and executing...")
                        plan, solver_error = self._call_pddl_solver(domain, problem, verbose=verbose)
                        solver_result = f"PDDL Solver Result:\n" + "\n".join(f"{i+1}. {a}" for i, a in enumerate(plan)) if plan else f"PDDL Solver Error:\n{solver_error or 'No plan found'}"
                        self._log_message(player_clients, current_player, "user", solver_result, run_dir)
                        if plan is None or len(plan) == 0:
                            feedback = f"ERROR: PDDL solver failed for your plan. {solver_error or 'No plan found'}."
                            if retry < max_pddl_retries - 1:
                                continue
                            break
                        feedback = None
                        plan_index = 0
                        current_valid_actions = valid_actions
                        executed_any = False
                        while plan_index < len(plan):
                            plan_action = plan[plan_index]
                            if verbose:
                                print(f"  Parsing plan action {plan_index + 1}/{len(plan)}: '{plan_action}'")
                            parsed_action = self._parse_plan_action(plan_action, current_valid_actions)
                            if parsed_action is None or parsed_action not in current_valid_actions:
                                if not executed_any:
                                    feedback = f"ERROR: First action '{plan_action}' invalid. Valid: {', '.join(current_valid_actions)}."
                                    if retry < max_pddl_retries - 1:
                                        break
                                else:
                                    if verbose:
                                        print(f"  Stopping plan execution after {plan_index} actions.")
                                break
                            acting_player = game.current_player
                            obs, reward, done, info = game.step(parsed_action)
                            executed_any = True
                            stats.total_turns = turn_num
                            stats.player_turns.append(acting_player)
                            stats.player_actions[acting_player].append(parsed_action)
                            stats.final_score = info['scoreNormalized']
                            result_msg = f"Turn {turn_num} - Result of executing '{parsed_action}':\n{obs}\n\nReward: {reward}. Score: {info['scoreNormalized']:.2f}."
                            self._log_message(player_clients, acting_player, "user", result_msg, run_dir)
                            self._log_message(player_clients, acting_player, "assistant", f"Executed: {parsed_action}", run_dir)
                            if verbose:
                                print(f"  Turn {turn_num} - Executed: {parsed_action}")
                            if done:
                                stats.game_won = info['taskSuccess']
                                break
                            turn_num += 1
                            if max_turns and turn_num > max_turns:
                                break
                            plan_index += 1
                            current_valid_actions = [act[0] for act in game.last_valid_actions]
                            if not current_valid_actions:
                                break
                        if executed_any:
                            break
                        if retry >= max_pddl_retries - 1:
                            break
                    if not executed_any:
                        if verbose:
                            print(f"Warning: Failed to get valid action after {max_pddl_retries} two-player PDDL attempts. Falling back to standard action extraction.")
                        action = self._extract_action(llm_response, valid_actions)
                else:
                    for retry in range(max_pddl_retries):
                        prompt = self._create_prompt(observation, valid_actions, current_player, turn_num, num_coins, use_pddl=True, feedback=feedback)
                    
                    if verbose and retry == 0:
                        print(f"\n--- Turn {turn_num} - Player {current_player + 1} (PDDL Mode) ---")
                        print(f"Observation: {observation[:200]}..." if len(observation) > 200 else f"Observation: {observation}")
                    
                    llm_response = self._call_llm(prompt, current_player, player_clients, system_prompt, run_dir)
                    llm_response_cleaned = self._strip_reasoning_tokens(llm_response) if llm_response else ""
                    
                    if verbose:
                        if llm_response:
                            print(f"LLM Response (PDDL attempt {retry + 1}): {llm_response[:500]}..." if len(llm_response) > 500 else f"LLM Response (PDDL attempt {retry + 1}): {llm_response}")
                        else:
                            print(f"LLM Response (PDDL attempt {retry + 1}): [BLANK/EMPTY]")
                    
                    domain, problem = self._extract_pddl_files(llm_response_cleaned)
                    
                    if not domain or not problem:
                        missing = []
                        if not domain:
                            missing.append("domain")
                        if not problem:
                            missing.append("problem")
                        feedback = f"ERROR: Could not extract PDDL {', '.join(missing)} file(s) from your response. Please provide both domain and problem files in the specified format. Make sure to include:\n- ```pddl\n(define (domain ...)\n  ...\n)\n```\n- ```pddl\n(define (problem ...)\n  ...\n)\n```"
                        if verbose:
                            print(f"Warning: Failed to extract PDDL files (attempt {retry + 1}/{max_pddl_retries})")
                            print(f"  Domain found: {domain is not None}")
                            print(f"  Problem found: {problem is not None}")
                            if verbose and llm_response:
                                print(f"  Response preview (first 500 chars): {llm_response[:500]}")
                        if retry < max_pddl_retries - 1:
                            continue
                        else:
                            break
                    
                    if verbose:
                        print(f"Extracted PDDL domain ({len(domain)} chars) and problem ({len(problem)} chars)")
                    
                    if verbose:
                        print(f"Validating PDDL syntax...")
                    pddl_debug_dir = run_dir / 'pddl_debug' if run_dir else None
                    is_valid, syntax_error = self._validate_pddl_syntax(domain, problem, verbose=verbose, save_dir=pddl_debug_dir, turn_num=turn_num, attempt=retry+1)
                    
                    if not is_valid:
                        feedback = f"ERROR: PDDL syntax validation failed. {syntax_error}. Please check your domain and problem files for syntax errors like missing parentheses, undefined predicates/actions, or invalid PDDL structure."
                        if verbose:
                            print(f"  PDDL syntax error: {syntax_error}")
                            if pddl_debug_dir:
                                print(f"  Full PDDL files saved to: {pddl_debug_dir}")
                            else:
                                print(f"  Domain preview: {domain[:300]}...")
                                print(f"  Problem preview: {problem[:300]}...")
                        if retry < max_pddl_retries - 1:
                            continue
                        else:
                            break
                    
                    if pddl_debug_dir and verbose:
                        domain_save_path = pddl_debug_dir / f"domain_turn{turn_num}_attempt{retry+1}_valid.pddl"
                        problem_save_path = pddl_debug_dir / f"problem_turn{turn_num}_attempt{retry+1}_valid.pddl"
                        with open(domain_save_path, 'w') as f:
                            f.write(domain)
                        with open(problem_save_path, 'w') as f:
                            f.write(problem)
                        if verbose:
                            print(f"  PDDL files saved: {domain_save_path.name}, {problem_save_path.name}")
                    
                    if verbose:
                        print(f"Calling PDDL solver with domain ({len(domain)} chars) and problem ({len(problem)} chars)...")
                    
                    debug_dir = None
                    if verbose:
                        import tempfile
                        debug_dir = tempfile.mkdtemp(prefix='pddl_debug_')
                        domain_debug_file = os.path.join(debug_dir, f'domain_turn{turn_num}_attempt{retry+1}.pddl')
                        problem_debug_file = os.path.join(debug_dir, f'problem_turn{turn_num}_attempt{retry+1}.pddl')
                        with open(domain_debug_file, 'w') as f:
                            f.write(domain)
                        with open(problem_debug_file, 'w') as f:
                            f.write(problem)
                        if verbose:
                            print(f"  Debug: Domain saved to {domain_debug_file}")
                            print(f"  Debug: Problem saved to {problem_debug_file}")
                    
                    plan, solver_error = self._call_pddl_solver(domain, problem, verbose=verbose)
                    solver_result = f"PDDL Solver Result:\n" + "\n".join(f"{i+1}. {a}" for i, a in enumerate(plan)) if plan else f"PDDL Solver Error:\n{solver_error or 'No plan found'}"
                    self._log_message(player_clients, current_player, "user", solver_result, run_dir)
                    
                    if plan is None or len(plan) == 0:
                        error_details = []
                        if solver_error:
                            error_details.append(f"Solver error: {solver_error[:500]}")
                        
                        error_details.append(f"Domain length: {len(domain)} chars, Problem length: {len(problem)} chars")
                        if debug_dir:
                            error_details.append(f"Domain and problem files saved to: {debug_dir}")
                        
                        feedback = f"ERROR: PDDL solver failed or returned no plan. {' '.join(error_details)}. Please check your domain and problem files for syntax errors. Common issues: missing parentheses, undefined predicates/actions, type mismatches, or invalid PDDL syntax."
                        
                        if verbose:
                            print(f"Warning: PDDL solver failed (attempt {retry + 1}/{max_pddl_retries})")
                            print(f"  Domain preview (first 500 chars):\n{domain[:500]}")
                            print(f"  Problem preview (first 500 chars):\n{problem[:500]}")
                            if solver_error:
                                print(f"  Solver error output:\n{solver_error[:1000]}")
                        if retry < max_pddl_retries - 1:
                            continue
                        else:
                            break
                    
                    if verbose:
                        print(f"PDDL Plan received: {plan[:3]}..." if len(plan) > 3 else f"PDDL Plan: {plan}")
                    
                    if not plan:
                        feedback = "ERROR: Plan is empty. Please check your PDDL domain and problem files."
                        if verbose:
                            print(f"Warning: Empty plan (attempt {retry + 1}/{max_pddl_retries})")
                        if retry < max_pddl_retries - 1:
                            continue
                        else:
                            break
                    
                    plan_index = 0
                    current_valid_actions = valid_actions
                    executed_any = False
                    
                    while plan_index < len(plan):
                        plan_action = plan[plan_index]
                        if verbose:
                            print(f"  Parsing plan action {plan_index + 1}/{len(plan)}: '{plan_action}'")
                        parsed_action = self._parse_plan_action(plan_action, current_valid_actions)
                        if parsed_action is None or parsed_action not in current_valid_actions:
                            if not executed_any:
                                feedback = f"ERROR: The first action in the plan '{plan_action}' could not be converted to a valid game action. The PDDL plan action format is fine, but it needs to match game actions like 'move north', 'move south', 'take coin1', etc. Valid actions are: {', '.join(current_valid_actions)}."
                                if verbose:
                                    print(f"  Invalid or not in valid actions. Stopping. Parsed: '{parsed_action}', Valid: {current_valid_actions}")
                                    print(f"Warning: Invalid action '{plan_action}' (attempt {retry + 1}/{max_pddl_retries})")
                                if retry < max_pddl_retries - 1:
                                    break
                                else:
                                    break
                            else:
                                if verbose:
                                    print(f"  Plan action '{plan_action}' invalid or not in valid actions. Stopping plan execution after {plan_index} actions.")
                                break
                        
                        acting_player = game.current_player
                        obs, reward, done, info = game.step(parsed_action)
                        executed_any = True
                        stats.total_turns = turn_num
                        stats.player_turns.append(acting_player)
                        stats.player_actions[acting_player].append(parsed_action)
                        stats.final_score = info['scoreNormalized']
                        
                        result_msg = f"Turn {turn_num} - Result of executing '{parsed_action}':\n{obs}\n\nReward: {reward}. Score: {info['scoreNormalized']:.2f}."
                        self._log_message(player_clients, acting_player, "user", result_msg, run_dir)
                        self._log_message(player_clients, acting_player, "assistant", f"Executed: {parsed_action}", run_dir)
                        
                        if verbose:
                            print(f"  Turn {turn_num} - Executed: {parsed_action}")
                            print(f"  Reward: {reward}, Score: {info['scoreNormalized']:.2f} ({info['scoreRaw']}/{len(game.task_objects)})")
                        
                        if done:
                            stats.game_won = info['taskSuccess']
                            if verbose:
                                if stats.game_won:
                                    print(f"\nGame Won! All players collected all coins together!")
                                else:
                                    print(f"\nGame ended (step limit reached)")
                            break
                        
                        turn_num += 1
                        if max_turns and turn_num > max_turns:
                            if verbose:
                                print(f"  Max turns ({max_turns}) reached. Stopping plan execution.")
                            break
                        
                        plan_index += 1
                        current_valid_actions = [act[0] for act in game.last_valid_actions]
                        if not current_valid_actions:
                            if verbose:
                                print(f"  No valid actions available. Stopping plan execution.")
                            break
                    
                    if executed_any:
                        break
                    if retry < max_pddl_retries - 1:
                        continue
                    else:
                        break
                
                if not executed_any:
                    if verbose:
                        print(f"Warning: Failed to get valid action after {max_pddl_retries} PDDL attempts. Falling back to standard action extraction.")
                    action = self._extract_action(llm_response, valid_actions)
            else:
                prompt = self._create_prompt(observation, valid_actions, current_player, turn_num, num_coins, use_pddl=False)
                
                if verbose:
                    print(f"\n--- Turn {turn_num} - Player {current_player + 1} ---")
                    print(f"Observation: {observation[:200]}..." if len(observation) > 200 else f"Observation: {observation}")
                
                llm_response = self._call_llm(prompt, current_player, player_clients, system_prompt, run_dir)
                
                if verbose:
                    if llm_response:
                        print(f"LLM Response: {llm_response}")
                    else:
                        print(f"LLM Response: [BLANK/EMPTY]")
                
                action = self._extract_action(llm_response, valid_actions)
            
            if use_pddl_mode and executed_any:
                continue
            
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
        
        if run_dir:
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


def generate_config_combinations(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    sweep_params = {}
    fixed_params = {}
    
    for key, value in config.items():
        if isinstance(value, list) and key not in ['api_params']:
            sweep_params[key] = value
        else:
            fixed_params[key] = value
    
    if not sweep_params:
        return [config]
    
    list_lengths = [len(v) for v in sweep_params.values()]
    if len(set(list_lengths)) > 1:
        raise ValueError(
            f"All list parameters must have the same length. Found lengths: {dict(zip(sweep_params.keys(), list_lengths))}"
        )
    
    n_experiments = list_lengths[0] if list_lengths else 0
    keys = list(sweep_params.keys())
    combinations = []
    
    for i in range(n_experiments):
        combo_config = fixed_params.copy()
        for key in keys:
            combo_config[key] = sweep_params[key][i]
        combinations.append(combo_config)
    
    return combinations


def call_visualize_experiment(experiment_dir: Path) -> bool:
    try:
        script_path = Path(__file__).parent / "visualize_experiment.py"
        result = subprocess.run(
            [sys.executable, str(script_path), str(experiment_dir)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"Warning: Visualization failed for {experiment_dir}: {result.stderr}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"Warning: Error calling visualization for {experiment_dir}: {e}", file=sys.stderr)
        return False


def _create_run_suffix(config: Dict[str, Any], base_config: Dict[str, Any], combo_idx: int, total_combos: int) -> str:
    if total_combos == 1:
        return ""
    
    varying_params = []
    for key, value in base_config.items():
        if isinstance(value, list) and key not in ['api_params']:
            varying_params.append(key)
    
    if not varying_params:
        return f"run{combo_idx + 1}"
    
    parts = []
    for param in varying_params:
        value = config[param]
        if isinstance(value, bool):
            short_val = "T" if value else "F"
        elif isinstance(value, (int, float)):
            short_val = str(value)
        elif isinstance(value, str):
            short_val = value[:4].replace(' ', '_')
        else:
            short_val = str(value)[:4]
        parts.append(f"{param[:3]}{short_val}")
    
    suffix = "_".join(parts)
    suffix = ''.join(c if c.isalnum() or c in ('_', '-', '.') else '_' for c in suffix)
    return f"run{combo_idx + 1}_{suffix}"


def _parse_config_value(key: str, value: Any) -> Any:
    if key in ['include_doors', 'coins_in_containers', 'limit_inventory_size']:
        if isinstance(value, str):
            value_str = value.lower()
            return value_str in ('true', '1', 'yes', 'on')
        return bool(value)
    elif key == 'connectivity':
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.5
        return float(value) if value is not None else 0.5
    elif key in ['num_locations', 'num_coins', 'num_players', 'num_distractor_items', 'max_steps', 'seed']:
        return value
    return value


def main():
    parser = argparse.ArgumentParser(description='Run Coin Collector game with LLM agents using VLLM')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to YAML config file (default: config.yaml)')
    
    args = parser.parse_args()
    
    load_env_file('.env')
    base_config = load_config(args.config)
    
    config_combinations = generate_config_combinations(base_config)
    total_runs = len(config_combinations)
    
    print(f"\n=== Running {total_runs} experiment(s) ===")
    if total_runs > 1:
        print("Multiple parameter combinations detected. Running each sequentially.\n")
    
    exit_code = 0
    
    for combo_idx, config in enumerate(config_combinations):
        if total_runs > 1:
            print(f"\n{'='*60}")
            print(f"Experiment {combo_idx + 1} of {total_runs}")
            print(f"{'='*60}\n")
        
        model = config.get('model', 'vllm')
        model_name = config.get('model_name', 'Qwen/Qwen3-32B')
        api_params = config.get('api_params', {})
        num_locations = config.get('num_locations', 20)
        num_coins = config.get('num_coins', 3)
        num_players = config.get('num_players', 2)
        max_steps = config.get('max_steps')
        seed = config.get('seed')
        
        include_doors = _parse_config_value('include_doors', config.get('include_doors', True))
        num_distractor_items = config.get('num_distractor_items', 0)
        coins_in_containers = _parse_config_value('coins_in_containers', config.get('coins_in_containers', False))
        limit_inventory_size = _parse_config_value('limit_inventory_size', config.get('limit_inventory_size', True))
        connectivity = _parse_config_value('connectivity', config.get('connectivity', 0.5))
        
        quiet = config.get('quiet', False)
        output_file = config.get('output')
        no_auto_save = config.get('no_auto_save', False)
        prompt_version = config.get('prompt_version', 'standard')
        use_pddl = config.get('use_pddl', False)
        if isinstance(use_pddl, str):
            use_pddl = use_pddl.lower() in ('true', '1', 'yes', 'on')
        
        if use_pddl and 'Qwen3-32B' not in model_name:
            print(f"Warning: use_pddl is enabled but model '{model_name}' is not Qwen3-32B. PDDL mode only works with Qwen3-32B. Disabling PDDL mode.")
            use_pddl = False
        
        run_suffix = _create_run_suffix(config, base_config, combo_idx, total_runs)
        
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
                prompt_version=prompt_version,
                use_pddl=use_pddl,
                verbose=not quiet,
                auto_save=not no_auto_save,
                output_file=output_file,
                run_dir_suffix=run_suffix
            )
            
            if run_dir:
                print(f"\nGenerating visualizations for {run_dir}...")
                call_visualize_experiment(run_dir)
            
            if not stats.game_won:
                exit_code = 1
            
        except Exception as e:
            print(f"Error in experiment {combo_idx + 1}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            exit_code = 1
    
    if total_runs > 1:
        print(f"\n{'='*60}")
        print(f"Completed {total_runs} experiment(s)")
        print(f"{'='*60}\n")
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
