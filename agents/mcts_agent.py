# agents/mcts_agent.py
import torch
from models.selfplay import MCTSWithNN
from models.net import SimpleNet
from models.encoder import StateEncoder
from simulator.engine import legal_actions
from typing import Tuple, Dict, Any

class MCTSAgent:
    """Agent using MCTS guided by neural network"""
    
    def __init__(self, playouts: int = 50, c_puct: float = 1.0):
        self.playouts = playouts
        self.c_puct = c_puct
        self.net = None
        self.encoder = None
        self.mcts = None
    
    def _init_net(self, max_card_id: int):
        """Lazy initialization of network"""
        if self.net is None:
            self.net = SimpleNet(state_dim=256, action_dim=4096)
            self.encoder = StateEncoder(max_card_id, state_dim=256)
            self.mcts = MCTSWithNN(
                self.net, self.encoder, max_card_id,
                c_puct=self.c_puct, playouts=self.playouts, device='cpu'
            )
    
    def select_action(self, state) -> Tuple[str, Dict[str, Any]]:
        """Select action using MCTS"""
        # Fallback to random if network not initialized
        legal_acts = legal_actions(state)
        if not legal_acts:
            return ("pass", {})
        
        try:
            if self.mcts is not None:
                action, _, _ = self.mcts.run(state)
                return action
        except Exception as e:
            print(f"MCTS error: {e}, falling back to random")
        
        import random
        return random.choice(legal_acts)
