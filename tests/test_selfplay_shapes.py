# tests/test_selfplay_shapes.py
import pytest
import torch
from simulator.card_loader import load_cards
from models.net import SimpleNet
from models.encoder import StateEncoder
from models.selfplay import self_play_episode
from run.run_match import build_deck_from_names

def test_selfplay_policy_shape():
    card_db = load_cards()
    max_card_id = max(card_db.keys())
    encoder = StateEncoder(max_card_id, state_dim=256)
    net = SimpleNet(state_dim=256, action_dim=4096)
    deck_names = ["Greninja ex", "Magcargo ex", "Incineroar ex"]
    deck1 = build_deck_from_names(card_db, deck_names)
    deck2 = build_deck_from_names(card_db, deck_names)
    examples = self_play_episode(card_db, deck1, deck2, net, encoder, {'c_puct':1.0, 'playouts':10}, device='cpu')
    assert len(examples) > 0
    s_t, pol_vec, val = examples[0]
    assert hasattr(s_t, "shape")
    assert len(pol_vec) == 4096
