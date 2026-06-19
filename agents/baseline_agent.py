# agents/baseline_agent.py
from simulator.engine import legal_actions
from typing import Tuple, Dict, Any

class BaselineAgent:
    """Baseline agent that follows a simple heuristic strategy"""
    
    def __init__(self, name: str = "Baseline"):
        self.name = name
    
    def select_action(self, state) -> Tuple[str, Dict[str, Any]]:
        """
        Select action using a simple heuristic:
        1. Attack if possible
        2. Bench a pokemon if hand is full
        3. Play trainers
        4. Attach energy
        5. Pass
        """
        actions = legal_actions(state)
        
        if not actions:
            return ("pass", {})
        
        # Priority: attack > bench > play_trainer > attach > pass
        for action_type in ["attack", "bench", "play_trainer", "attach"]:
            for a in actions:
                if a[0] == action_type:
                    return a
        
        # Default to pass
        return ("pass", {})
