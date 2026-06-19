# tests/test_action_space.py
import pytest
from models.action_space import init_action_space, encode_attach_active, decode_index, ACTION_DIM

def test_action_space_roundtrip():
    init_action_space(max_card_id=500, max_bench=5, max_moves=4, action_dim=2048)
    idx = encode_attach_active(42)
    action = decode_index(idx)
    assert action[0] == "attach"
    assert action[1]["card_id"] == 42
    assert ACTION_DIM() >= 1
