# models/action_space.py
"""
Canonical action index mapping.

Usage:
  - Call init_action_space(max_card_id) once after loading cards.json
  - Use encode_* helpers to map action payloads to indices
  - Use legal_actions_to_mask(legal_actions) to build mask for network
  - Use decode_index(idx) to map index back to an action tuple
"""

from typing import Tuple, Dict, Any, List
import math

# Globals set by init_action_space
_MAX_CARD_ID = 2000
_MAX_BENCH = 5
_MAX_MOVES = 4

# Offsets (computed in init)
_OFFSET_ATTACH_ACTIVE = 0
_OFFSET_ATTACH_BENCH = 0
_OFFSET_BENCH = 0
_OFFSET_ATTACK = 0
_OFFSET_RETREAT = 0
_OFFSET_PLAY_TRAINER = 0
_OFFSET_PASS = 0
_ACTION_DIM = 4096  # default fallback

def init_action_space(max_card_id: int, max_bench: int = 5, max_moves: int = 4, action_dim: int = None):
    global _MAX_CARD_ID, _MAX_BENCH, _MAX_MOVES
    global _OFFSET_ATTACH_ACTIVE, _OFFSET_ATTACH_BENCH, _OFFSET_BENCH, _OFFSET_ATTACK
    global _OFFSET_RETREAT, _OFFSET_PLAY_TRAINER, _OFFSET_PASS, _ACTION_DIM
    _MAX_CARD_ID = max_card_id
    _MAX_BENCH = max_bench
    _MAX_MOVES = max_moves
    BLOCK_ATTACH_ACTIVE = _MAX_CARD_ID + 1
    BLOCK_ATTACH_BENCH = (_MAX_CARD_ID + 1) * _MAX_BENCH
    BLOCK_BENCH = _MAX_CARD_ID + 1
    BLOCK_ATTACK = _MAX_MOVES
    BLOCK_RETREAT = 1
    BLOCK_PLAY_TRAINER = _MAX_CARD_ID + 1
    BLOCK_PASS = 1
    _OFFSET_ATTACH_ACTIVE = 0
    _OFFSET_ATTACH_BENCH = _OFFSET_ATTACH_ACTIVE + BLOCK_ATTACH_ACTIVE
    _OFFSET_BENCH = _OFFSET_ATTACH_BENCH + BLOCK_ATTACH_BENCH
    _OFFSET_ATTACK = _OFFSET_BENCH + BLOCK_BENCH
    _OFFSET_RETREAT = _OFFSET_ATTACK + BLOCK_ATTACK
    _OFFSET_PLAY_TRAINER = _OFFSET_RETREAT + BLOCK_RETREAT
    _OFFSET_PASS = _OFFSET_PLAY_TRAINER + BLOCK_PLAY_TRAINER
    computed = _OFFSET_PASS + BLOCK_PASS
    _ACTION_DIM = action_dim if action_dim is not None else max(4096, computed)

def ACTION_DIM() -> int:
    return _ACTION_DIM

def encode_attach_active(card_id: int) -> int:
    return _OFFSET_ATTACH_ACTIVE + card_id

def encode_attach_bench(card_id: int, bench_slot: int) -> int:
    return _OFFSET_ATTACH_BENCH + bench_slot * (_MAX_CARD_ID + 1) + card_id

def encode_bench(card_id: int) -> int:
    return _OFFSET_BENCH + card_id

def encode_attack(move_index: int) -> int:
    if move_index < 0 or move_index >= _MAX_MOVES:
        raise ValueError("move_index out of range")
    return _OFFSET_ATTACK + move_index

def encode_retreat() -> int:
    return _OFFSET_RETREAT

def encode_play_trainer(card_id: int) -> int:
    return _OFFSET_PLAY_TRAINER + card_id

def encode_pass() -> int:
    return _OFFSET_PASS

def decode_index(idx: int) -> Tuple[str, Dict[str, Any]]:
    if idx < _OFFSET_ATTACH_BENCH:
        card_id = idx - _OFFSET_ATTACH_ACTIVE
        return ("attach", {"card_id": card_id, "target": "active"})
    if idx < _OFFSET_BENCH:
        rel = idx - _OFFSET_ATTACH_BENCH
        bench_slot = rel // (_MAX_CARD_ID + 1)
        card_id = rel % (_MAX_CARD_ID + 1)
        return ("attach", {"card_id": card_id, "target": "bench", "bench_slot": bench_slot})
    if idx < _OFFSET_ATTACK:
        card_id = idx - _OFFSET_BENCH
        return ("bench", {"card_id": card_id})
    if idx < _OFFSET_RETREAT:
        move_index = idx - _OFFSET_ATTACK
        return ("attack", {"move_index": move_index})
    if idx < _OFFSET_PLAY_TRAINER:
        return ("retreat", {})
    if idx < _OFFSET_PASS:
        card_id = idx - _OFFSET_PLAY_TRAINER
        return ("play_trainer", {"card_id": card_id})
    if idx < _ACTION_DIM:
        return ("pass", {})
    raise ValueError("index out of action space")

def _action_to_index(action: Tuple[str, Dict[str, Any]]) -> int:
    a, payload = action
    if a == "attach":
        card_id = payload.get("card_id", 0)
        target = payload.get("target", "active")
        if target == "active":
            return encode_attach_active(card_id)
        else:
            bench_slot = payload.get("bench_slot", 0)
            return encode_attach_bench(card_id, bench_slot)
    if a == "bench":
        return encode_bench(payload.get("card_id", 0))
    if a == "attack":
        return encode_attack(payload.get("move_index", 0))
    if a == "retreat":
        return encode_retreat()
    if a == "play_trainer":
        return encode_play_trainer(payload.get("card_id", 0))
    if a == "pass":
        return encode_pass()
    return None

def legal_actions_to_mask(legal_actions: List[Tuple[str, Dict[str, Any]]]) -> List[float]:
    mask = [0.0] * _ACTION_DIM
    for a, payload in legal_actions:
        try:
            idx = _action_to_index((a, payload))
            if idx is not None and 0 <= idx < _ACTION_DIM:
                mask[idx] = 1.0
        except Exception:
            continue
    return mask

def policy_to_action(policy_vector: List[float], legal_mask: List[float]) -> Tuple[str, Dict[str, Any]]:
    best_idx = None
    best_val = -float("inf")
    for i, (p, m) in enumerate(zip(policy_vector, legal_mask)):
        if m > 0 and p > best_val:
            best_val = p
            best_idx = i
    if best_idx is None:
        return ("pass", {})
    return decode_index(best_idx)
