# agents/random_agent.py
import random
from simulator.engine import legal_actions
from typing import Tuple, Dict, Any

class RandomAgent:
    """Agent that selects random legal actions"""
    
    def __init__(self, name: str = "Random"):
        self.name = name
    
    def select_action(self, state) -> Tuple[str, Dict[str, Any]]:
        """Select a random legal action"""
        actions = legal_actions(state)
        
        if not actions:
            return ("pass", {})
        
        return random.choice(actions)
