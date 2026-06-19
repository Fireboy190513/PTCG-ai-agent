# simulator/state.py
from typing import List, Dict, Any, Optional

class CardInstance:
    def __init__(self, card_def: Dict[str, Any]):
        self.defn = card_def
        self.name = card_def.get("name", "Unknown")
        self.current_hp = card_def.get("hp", 1)
        self.attached = []  # list of CardInstance (energy)

class PlayerState:
    def __init__(self, deck: List[Dict[str, Any]], name: str = "Player"):
        self.name = name
        self.deck = [card_def for card_def in deck]
        self.hand = []
        self.bench = []
        self.active: Optional[CardInstance] = None
        self.discard = []
        self.prizes = []
        self.took_prize = 0

    def draw(self, count: int):
        for _ in range(count):
            if self.deck:
                card_def = self.deck.pop(0)
                self.hand.append(CardInstance(card_def))

class GameState:
    def __init__(self, p1: PlayerState, p2: PlayerState):
        self.players = [p1, p2]
        self.current = 0  # current player index
        self.turn = 0
        self.log = []

    def clone(self):
        import copy
        return copy.deepcopy(self)
