# simulator/serialize.py
from collections import Counter
from simulator.state import GameState, CardInstance
from typing import Dict, Any, List

# Map energy/type strings to indices (adjust to your CSV types)
ENERGY_MAP = {"{G}":0, "{R}":1, "{W}":2, "{L}":3, "{P}":4, "{F}":5, "{D}":6, "{M}":7, "{C}":8}
TYPE_COUNT = len(ENERGY_MAP)
MAX_HAND_SUMMARY = 128

def energy_type_to_index(t: str):
    return ENERGY_MAP.get(t, None)

def card_instance_to_def(card_inst: CardInstance) -> Dict[str, Any]:
    if card_inst is None:
        return None
    energy_counts = [0]*TYPE_COUNT
    for e in getattr(card_inst, "attached", []):
        t = e.defn.get("Type", "")
        idx = energy_type_to_index(t)
        if idx is not None and idx < TYPE_COUNT:
            energy_counts[idx] += 1
    return {
        "id": card_inst.defn.get("id", 0),
        "hp": card_inst.defn.get("hp", 1) or 1,
        "current_hp": getattr(card_inst, "current_hp", card_inst.defn.get("hp", 1)) or 1,
        "retreat": card_inst.defn.get("retreat", 0) or 0,
        "attached_energy_counts": energy_counts,
        "status_mask": 0
    }

def serialize_state(game_state: GameState) -> Dict[str, Any]:
    cur = game_state
    p = cur.players[cur.current]
    opp = cur.players[1 - cur.current]
    s = {}
    s["active"] = card_instance_to_def(p.active)
    s["bench"] = [card_instance_to_def(b) for b in p.bench]
    # hand summary: histogram of card ids truncated/padded to MAX_HAND_SUMMARY
    hand_ids = [c.defn.get("id", 0) for c in p.hand]
    cnt = Counter(hand_ids)
    hand_vec = [0.0] * MAX_HAND_SUMMARY
    i = 0
    for k, v in cnt.items():
        if i >= MAX_HAND_SUMMARY:
            break
        hand_vec[i] = float(v) / 4.0
        i += 1
    s["hand_summary"] = hand_vec
    s["global"] = {
        "turn": cur.turn,
        "current_player": cur.current,
        "p1_prizes": len(cur.players[0].prizes),
        "p2_prizes": len(cur.players[1].prizes),
        "p1_deck": len(cur.players[0].deck),
        "p2_deck": len(cur.players[1].deck),
        "p1_discard": len(cur.players[0].discard),
        "p2_discard": len(cur.players[1].discard),
        "stadium_id": 0
    }
    return s
