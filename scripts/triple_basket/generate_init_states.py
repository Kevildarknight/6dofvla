"""Create data/init_states/*.pruned_init: one fixed, settled start state (robot + 3 items + basket).

Usage: .venv/bin/python scripts/triple_basket/generate_init_states.py
"""
import torch
from common import DUMMY, make_env, tb

env = make_env()
env.reset()
for _ in range(20):  # let objects settle on the floor
    env.step(DUMMY)
state = env.env.sim.get_state().flatten()
tb.INIT_DIR.mkdir(parents=True, exist_ok=True)
torch.save(state[None, :], tb.INIT_FILE)
print(f"saved {tb.INIT_FILE} with state dim {state.shape[0]}")
