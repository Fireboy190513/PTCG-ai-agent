# models/trainer.py
import torch
import torch.nn.functional as F
import os
from models.net import SimpleNet
from models.encoder import StateEncoder
from models.replay_buffer import ReplayBuffer
from models.selfplay import self_play_episode
from simulator.card_loader import load_cards
from run.run_match import build_deck_from_names
import random

def train_loop(episodes=100, selfplay_per_episode=1, device='cpu'):
    card_db = load_cards()
    max_card_id = max(card_db.keys())
    encoder = StateEncoder(max_card_id, state_dim=256)
    net = SimpleNet(state_dim=256, action_dim=4096).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    buffer = ReplayBuffer(capacity=200000)
    mcts_cfg = {'c_puct': 1.0, 'playouts': 50}
    for ep in range(episodes):
        for _ in range(selfplay_per_episode):
            deck_names = ["Greninja ex", "Magcargo ex", "Incineroar ex"]
            deck1 = build_deck_from_names(card_db, deck_names)
            deck2 = build_deck_from_names(card_db, deck_names)
            examples = self_play_episode(card_db, deck1, deck2, net, encoder, mcts_cfg, device=device)
            for s_t, pol_vec, val in examples:
                buffer.push(s_t, pol_vec, val)
        if len(buffer) < 256:
            print(f"Buffer size {len(buffer)}; waiting for more examples")
            continue
        # training steps
        for it in range(100):
            states, policies, values = buffer.sample(64)
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)
            logits, pred_values = net(states)
            logp = torch.log_softmax(logits, dim=1)
            loss_p = - (policies * logp).sum(dim=1).mean()
            loss_v = F.mse_loss(pred_values.squeeze(-1), values.squeeze(-1))
            loss = loss_p + loss_v
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 5 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(net.state_dict(), f"models/checkpoint_ep{ep}.pt")
            print(f"Saved checkpoint_ep{ep}")
    print("Training finished")
