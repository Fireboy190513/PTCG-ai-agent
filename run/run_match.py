# run/run_match.py
import argparse, os
from simulator.card_loader import load_cards
from simulator.engine import setup_game, step, legal_actions
from agents.baseline_agent import BaselineAgent
from agents.mcts_agent import MCTSAgent
from agents.random_agent import RandomAgent

def build_deck_from_names(card_db, names, count=60):
    ids = []
    for n in names:
        for cid, c in card_db.items():
            if n.lower() in c["name"].lower():
                ids.append(cid)
                if len(ids) >= count:
                    return ids
    while len(ids) < count:
        ids.append(next(iter(card_db.keys())))
    return ids

def run_game(card_db, deck1_ids, deck2_ids, agent1, agent2, max_turns=200):
    state = setup_game(card_db, deck1_ids, deck2_ids)
    agents = [agent1, agent2]
    for t in range(max_turns):
        cur = state.current
        agent = agents[cur]
        action = agent.select_action(state)
        state, _, _, info = step(state, action)
        if info.get("err"):
            state, _, _, _ = step(state, ("pass", {}))
        for i, p in enumerate(state.players):
            if not p.prizes:
                return i
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", default="baseline")
    parser.add_argument("--p2", default="baseline")
    parser.add_argument("--games", type=int, default=10)
    args = parser.parse_args()

    card_db = load_cards()
    deck_names = ["Greninja ex", "Magcargo ex", "Incineroar ex"]
    deck1 = build_deck_from_names(card_db, deck_names)
    deck2 = build_deck_from_names(card_db, deck_names)

    def make_agent(name):
        if name == "baseline":
            return BaselineAgent(name)
        if name == "mcts":
            return MCTSAgent(playouts=50)
        return RandomAgent(name)

    a1 = make_agent(args.p1)
    a2 = make_agent(args.p2)

    wins = {0:0,1:0}
    for g in range(args.games):
        winner = run_game(card_db, deck1, deck2, a1, a2)
        if winner is not None:
            wins[winner] += 1
        print(f"Game {g+1} winner: {winner}")
    print("Wins:", wins)

if __name__ == "__main__":
    main()
