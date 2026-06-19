# models/encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.action_space import ACTION_DIM

# Simple encoder that maps serialized state dict to a fixed-size tensor
CARD_EMB_DIM = 64
MAX_BENCH = 5
TYPE_COUNT = 9
MAX_HAND_SUMMARY = 128

class CardEmbedding(nn.Module):
    def __init__(self, max_card_id: int, emb_dim: int = CARD_EMB_DIM):
        super().__init__()
        self.emb = nn.Embedding(max_card_id + 1, emb_dim)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, card_id_tensor):
        return self.emb(card_id_tensor)

class StateEncoder(nn.Module):
    def __init__(self, max_card_id: int, state_dim: int = 256):
        super().__init__()
        self.card_emb = CardEmbedding(max_card_id)
        self.card_emb_dim = CARD_EMB_DIM
        slot_in_dim = self.card_emb_dim + 4 + TYPE_COUNT
        self.slot_proj = nn.Sequential(
            nn.Linear(slot_in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.hand_proj = nn.Sequential(
            nn.Linear(MAX_HAND_SUMMARY, 128),
            nn.ReLU()
        )
        self.global_proj = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU()
        )
        total_dim = 128 * (1 + MAX_BENCH) + 128 + 128
        self.final_proj = nn.Sequential(
            nn.Linear(total_dim, state_dim),
            nn.ReLU()
        )

    def _encode_card_def(self, card_def: dict):
        # card_def: dict with id, hp, current_hp, retreat, attached_energy_counts, status_mask
        cid = torch.tensor([card_def.get("id", 0)], dtype=torch.long)
        emb = self.card_emb(cid).squeeze(0)
        hp = 0.0
        if card_def.get("hp"):
            hp = float(card_def.get("current_hp", card_def.get("hp", 1))) / max(1.0, float(card_def.get("hp", 1)))
        retreat = float(card_def.get("retreat", 0)) / 4.0
        status = float(card_def.get("status_mask", 0))
        energy_counts = card_def.get("attached_energy_counts", [0]*TYPE_COUNT)
        energy_tensor = torch.tensor(energy_counts, dtype=torch.float) / 10.0
        scalar = torch.tensor([hp, retreat, status, 0.0], dtype=torch.float)
        return torch.cat([emb, scalar, energy_tensor], dim=0)

    def forward(self, serialized_state):
        # serialized_state: dict (single) or list of dicts
        single = False
        if not isinstance(serialized_state, list):
            serialized_state = [serialized_state]
            single = True
        batch_slot_projs = []
        batch_hand = []
        batch_global = []
        for s in serialized_state:
            active = s.get("active")
            if active is None:
                active_def = {"id": 0, "hp": 1, "current_hp": 1, "retreat": 0, "attached_energy_counts": [0]*TYPE_COUNT, "status_mask": 0}
            else:
                active_def = active
            active_vec = self._encode_card_def(active_def)
            bench_defs = s.get("bench", [])
            bench_vecs = []
            for i in range(MAX_BENCH):
                if i < len(bench_defs) and bench_defs[i] is not None:
                    bdef = bench_defs[i]
                else:
                    bdef = {"id": 0, "hp": 1, "current_hp": 1, "retreat": 0, "attached_energy_counts": [0]*TYPE_COUNT, "status_mask": 0}
                bench_vecs.append(self._encode_card_def(bdef))
            slot_inputs = [active_vec] + bench_vecs
            slot_projs = [self.slot_proj(inp) for inp in slot_inputs]
            batch_slot_projs.append(torch.cat(slot_projs, dim=0))
            hand_vec = torch.tensor(s.get("hand_summary", [0.0]*MAX_HAND_SUMMARY), dtype=torch.float)
            batch_hand.append(self.hand_proj(hand_vec))
            g = s.get("global", {})
            gvec = torch.tensor([
                float(g.get("turn", 0))/100.0,
                float(g.get("current_player", 0)),
                float(g.get("p1_prizes", 0))/6.0,
                float(g.get("p2_prizes", 0))/6.0,
                float(g.get("p1_deck", 0))/60.0,
                float(g.get("p2_deck", 0))/60.0,
                float(g.get("p1_discard", 0))/60.0,
                float(g.get("p2_discard", 0))/60.0,
                float(g.get("stadium_id", 0))/100.0,
                0.0,0.0,0.0,0.0,0.0,0.0,0.0
            ], dtype=torch.float)
            batch_global.append(self.global_proj(gvec))
        slots_t = torch.stack(batch_slot_projs, dim=0)
        hand_t = torch.stack(batch_hand, dim=0)
        glob_t = torch.stack(batch_global, dim=0)
        concat = torch.cat([slots_t, hand_t, glob_t], dim=1)
        out = self.final_proj(concat)
        return out[0] if single else out
