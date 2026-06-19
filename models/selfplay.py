# models/selfplay.py
import math, numpy as np, torch, random
from copy import deepcopy
from simulator.serialize import serialize_state
from simulator.engine import legal_actions, step, setup_game
from models.action_space import init_action_space, ACTION_DIM, legal_actions_to_mask, decode_index
from typing import Tuple, List, Dict, Any

class PUCTNode:
    def __init__(self, state, parent=None, prior=1.0):
        self.state = state
        self.parent = parent
        self.children = {}  # action (tuple) -> node
        self.N = 0
        self.W = 0.0
        self.P = prior

    def Q(self):
        return self.W / self.N if self.N > 0 else 0.0

def uct_score(parent: PUCTNode, child: PUCTNode, c_puct: float):
    return child.Q() + c_puct * child.P * math.sqrt(parent.N) / (1 + child.N)

class MCTSWithNN:
    def __init__(self, net, encoder, card_db_max_id: int, c_puct=1.0, playouts=100, device='cpu'):
        init_action_space(card_db_max_id)
        self.net = net
        self.encoder = encoder
        self.c_puct = c_puct
        self.playouts = playouts
        self.device = device

    def _action_to_index(self, action: Tuple[str, Dict[str, Any]]):
        # use action_space._action_to_index via encode helpers
        from models.action_space import _action_to_index  # internal helper
        return _action_to_index(action)

    def run(self, root_state):
        root = PUCTNode(root_state.clone(), prior=1.0)
        for _ in range(self.playouts):
            node = root
            # selection
            while node.children:
                action, node = max(node.children.items(), key=lambda kv: uct_score(node, kv[1], self.c_puct))
            # expand
            actions = legal_actions(node.state)
            if not actions:
                continue
            # get network priors
            sdict = serialize_state(node.state)
            st_tensor = self.encoder(sdict).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, value = self.net(st_tensor)
                probs_all = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                v = float(value.cpu().numpy()[0, 0])
            # build legal mask and priors
            mask = np.array(legal_actions_to_mask(actions))
            priors = probs_all * mask
            s = priors.sum()
            if s <= 0:
                priors = mask
                s = priors.sum() or 1.0
            priors = priors / s
            # create children
            for a in actions:
                idx = self._action_to_index(a)
                if idx is None:
                    continue
                st_copy = node.state.clone()
                st_copy, _, _, _ = step(st_copy, a)
                node.children[a] = PUCTNode(st_copy, parent=node, prior=float(priors[idx]))
            # select a child to evaluate (random)
            if not node.children:
                continue
            action, child = random.choice(list(node.children.items()))
            leaf_value = v
            # backup
            cur = child
            while cur:
                cur.N += 1
                cur.W += leaf_value
                cur = cur.parent
        # build visit count vector
        from models.action_space import ACTION_DIM as _ACTION_DIM
        visits = np.zeros(_ACTION_DIM(), dtype=float)
        for a, n in root.children.items():
            idx = self._action_to_index(a)
            if idx is not None:
                visits[idx] = n.N
        total = visits.sum() or 1.0
        policy_vec = visits / total
        best_idx = int(visits.argmax())
        best_action = decode_index(best_idx)
        return best_action, policy_vec, None

def self_play_episode(card_db, deck1_ids, deck2_ids, net, encoder, mcts_cfg, device='cpu'):
    """
    Run one self-play episode using MCTSWithNN. Returns list of (state_tensor, policy_vec, value)
    """
    max_card_id = max(card_db.keys())
    mcts = MCTSWithNN(net, encoder, card_db_max_id=max_card_id, c_puct=mcts_cfg['c_puct'], playouts=mcts_cfg['playouts'], device=device)
    state = setup_game(card_db, deck1_ids, deck2_ids)
    examples = []
    max_turns = 300
    for t in range(max_turns):
        sdict = serialize_state(state)
        st_tensor = encoder(sdict).detach().cpu()
        action, policy_vec, _ = mcts.run(state)
        examples.append((st_tensor, policy_vec, None))
        state, _, _, info = step(state, action)
        if info.get("err"):
            state, _, _, _ = step(state, ("pass", {}))
        # terminal check
        for i, p in enumerate(state.players):
            if not p.prizes:
                winner = i
                final_examples = []
                for idx, (s_t, pol, _) in enumerate(examples):
                    # value from perspective of player to move when state was recorded
                    # approximate: if winner == state.current then +1 else -1
                    val = 1.0 if winner == state.current else -1.0
                    final_examples.append((s_t, pol, val))
                return final_examples
    # draw
    return [(s_t, pol, 0.0) for (s_t, pol, _) in examples]
