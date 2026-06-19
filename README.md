# PTCG AI Agent — Action Space + AlphaZero Integration

## Overview
This project implements:
- Canonical action indexing and masks
- Simulator that returns canonical payloads
- State serializer and encoder
- MCTS guided by a neural network (priors over full action space)
- Trainer using fixed-length policy targets (soft-target cross-entropy)

## Setup
1. Create virtualenv and install:
   pip install -r requirements.txt

2. Place `EN_Card_Data.csv` into `data/`.

3. Generate card DB:
   python simulator/card_loader.py

4. Run unit tests:
   pytest tests/test_action_space.py tests/test_selfplay_shapes.py

5. Quick baseline match:
   python run/run_match.py --p1 baseline --p2 baseline --games 10

6. Start a short training run:
   python -c "from models.trainer import train_loop; train_loop(episodes=10, selfplay_per_episode=1, device='cpu')"

## Notes
- After `simulator/card_loader.py` runs, the trainer and action space initialization use the max card id from the card DB.
- The simulator implements core mechanics; many complex card effects are not yet implemented and should be added to `simulator/engine.py`.
- The action space is initialized dynamically in MCTSWithNN and trainer.
