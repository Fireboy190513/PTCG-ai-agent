# simulator/engine.py
import random
from typing import List, Tuple, Dict, Any, Optional
from simulator.state import GameState, PlayerState, CardInstance
from models.action_space import (
    encode_attach_active, encode_attach_bench, encode_bench,
    encode_attack, encode_retreat, encode_play_trainer, encode_pass,
    legal_actions_to_mask, decode_index, ACTION_DIM
)

# Canonical action tuples returned by legal_actions:
# ("attach", {"card_id": int, "target":"active" or "bench", "bench_slot":int})
# ("bench", {"card_id": int})
# ("attack", {"move_index": int})
# ("retreat", {})
# ("play_trainer", {"card_id": int})
# ("pass", {})

def setup_game(card_db: Dict[int, dict], deck_ids_p1: List[int], deck_ids_p2: List[int]) -> GameState:
    def build_deck(ids):
        return [card_db[i] for i in ids]
    p1 = PlayerState(build_deck(deck_ids_p1), name="P1")
    p2 = PlayerState(build_deck(deck_ids_p2), name="P2")
    random.shuffle(p1.deck)
    random.shuffle(p2.deck)
    p1.draw(7)
    p2.draw(7)
    # prizes: store card defs (not instances) for simplicity
    p1.prizes = [p1.deck.pop(0) for _ in range(6)]
    p2.prizes = [p2.deck.pop(0) for _ in range(6)]
    # choose active: first Basic in hand else first card
    def choose_active(p: PlayerState):
        for i, c in enumerate(p.hand):
            if "Basic" in c.defn.get("category", ""):
                p.active = p.hand.pop(i)
                return
        if p.hand:
            p.active = p.hand.pop(0)
    choose_active(p1)
    choose_active(p2)
    return GameState(p1, p2)

def _hand_index_to_card_id(player: PlayerState, hand_index: int) -> Optional[int]:
    if hand_index < 0 or hand_index >= len(player.hand):
        return None
    return player.hand[hand_index].defn.get("id")

def legal_actions(state: GameState) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Return list of canonical actions for the current player.
    Each action payload includes card_id where relevant so action_space can index it.
    """
    p = state.players[state.current]
    actions = []
    # Attach energy from hand to active or bench
    for i, c in enumerate(p.hand):
        cat = c.defn.get("category", "")
        if "Energy" in cat:
            card_id = c.defn.get("id")
            if p.active:
                actions.append(("attach", {"card_id": card_id, "target": "active"}))
            # attach to each bench slot that exists
            for bidx in range(len(p.bench)):
                actions.append(("attach", {"card_id": card_id, "target": "bench", "bench_slot": bidx}))
    # Bench a Basic Pokémon from hand
    for i, c in enumerate(p.hand):
        if "Basic" in c.defn.get("category", ""):
            card_id = c.defn.get("id")
            if len(p.bench) < 5:
                actions.append(("bench", {"card_id": card_id}))
    # Play trainer/item/supporter from hand (approx: any non-energy, non-pokemon)
    for i, c in enumerate(p.hand):
        cat = c.defn.get("category", "")
        if "Trainer" in cat or ("Energy" not in cat and "Pokémon" not in cat):
            card_id = c.defn.get("id")
            actions.append(("play_trainer", {"card_id": card_id}))
    # Attack actions (use move indices)
    if p.active and p.active.defn.get("moves"):
        for midx, mv in enumerate(p.active.defn.get("moves", [])):
            actions.append(("attack", {"move_index": midx}))
    # Retreat (if active exists and retreat cost > 0 and bench has space)
    if p.active and len(p.bench) < 5:
        actions.append(("retreat", {}))
    # Pass / end turn
    actions.append(("pass", {}))
    return actions

def step(state: GameState, action: Tuple[str, Dict[str, Any]]) -> Tuple[GameState, int, bool, Dict[str, Any]]:
    """
    Apply canonical action to state. Returns (state, reward, done, info).
    Reward/done are minimal here; engine is deterministic and simple.
    """
    st = state
    p = st.players[st.current]
    opp = st.players[1 - st.current]
    atype, payload = action
    info = {}
    if atype == "attach":
        card_id = payload.get("card_id")
        target = payload.get("target")
        bench_slot = payload.get("bench_slot", 0)
        # find card in hand by id
        hand_idx = next((i for i, c in enumerate(p.hand) if c.defn.get("id") == card_id), None)
        if hand_idx is None:
            return st, -1, False, {"err": "attach: card not in hand"}
        card_inst = p.hand.pop(hand_idx)
        if target == "active":
            if not p.active:
                return st, -1, False, {"err": "attach: no active"}
            p.active.attached.append(card_inst)
            st.log.append(f"{p.name} attached {card_inst.name} to active")
        elif target == "bench":
            if bench_slot < 0 or bench_slot >= len(p.bench):
                # if bench slot doesn't exist, append if space
                if len(p.bench) < 5:
                    p.bench.append(card_inst)
                    st.log.append(f"{p.name} attached {card_inst.name} to new bench slot")
                else:
                    return st, -1, False, {"err": "attach: bench index invalid"}
            else:
                p.bench[bench_slot].attached.append(card_inst)
                st.log.append(f"{p.name} attached {card_inst.name} to bench {bench_slot}")
        else:
            return st, -1, False, {"err": "attach: unknown target"}
        return st, 0, False, {}
    if atype == "bench":
        card_id = payload.get("card_id")
        if len(p.bench) >= 5:
            return st, -1, False, {"err": "bench: bench full"}
        hand_idx = next((i for i, c in enumerate(p.hand) if c.defn.get("id") == card_id), None)
        if hand_idx is None:
            return st, -1, False, {"err": "bench: card not in hand"}
        card_inst = p.hand.pop(hand_idx)
        p.bench.append(card_inst)
        st.log.append(f"{p.name} benched {card_inst.name}")
        return st, 0, False, {}
    if atype == "play_trainer":
        card_id = payload.get("card_id")
        hand_idx = next((i for i, c in enumerate(p.hand) if c.defn.get("id") == card_id), None)
        if hand_idx is None:
            return st, -1, False, {"err": "play_trainer: card not in hand"}
        card_inst = p.hand.pop(hand_idx)
        # Simplified: Trainer effect not implemented; move to discard
        p.discard.append(card_inst)
        st.log.append(f"{p.name} played trainer {card_inst.name} (effect not implemented)")
        return st, 0, False, {}
    if atype == "attack":
        if not p.active:
            return st, -1, False, {"err": "attack: no active"}
        midx = payload.get("move_index", 0)
        moves = p.active.defn.get("moves", [])
        if midx < 0 or midx >= len(moves):
            return st, -1, False, {"err": "attack: invalid move index"}
        mv = moves[midx]
        dmg = 0
        try:
            dmg = int(mv.get("damage") or 0)
        except:
            dmg = 0
        if not opp.active:
            return st, -1, False, {"err": "attack: opponent has no active"}
        opp.active.current_hp -= dmg
        st.log.append(f"{p.name} used {mv.get('name')} for {dmg}")
        if opp.active.current_hp <= 0:
            st.log.append(f"{opp.name}'s {opp.active.name} knocked out")
            opp.discard.append(opp.active)
            opp.active = None
            # take a prize if available
            if p.prizes:
                prize = p.prizes.pop(0)
                # prize is a card def; convert to instance and add to hand
                p.hand.append(CardInstance(prize))
                p.took_prize += 1
                st.log.append(f"{p.name} took a prize")
            # promote bench to active if available
            if opp.bench:
                opp.active = opp.bench.pop(0)
        return st, 0, False, {}
    if atype == "retreat":
        # simplified: swap active with first bench if exists
        if not p.active:
            return st, -1, False, {"err": "retreat: no active"}
        if not p.bench:
            return st, -1, False, {"err": "retreat: no bench to switch"}
        p.bench.append(p.active)
        p.active = p.bench.pop(0)
        st.log.append(f"{p.name} retreated to {p.active.name}")
        return st, 0, False, {}
    if atype == "pass":
        # end turn: switch current player and draw
        st.current = 1 - st.current
        st.turn += 1
        st.players[st.current].draw(1)
        st.log.append(f"Turn {st.turn}: {st.players[st.current].name} drew 1")
        return st, 0, False, {}
    return st, -1, False, {"err": "unknown action"}
