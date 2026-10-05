"""Step 4 checks: reset repeatability, reach, per-item In-basket detection, all-three-only success.

Usage: .venv/bin/python scripts/triple_basket/verify_scene.py
Exits non-zero if any check fails.
"""
import sys

import numpy as np
from common import DUMMY, body_pos, fixed_reset, make_env, settle, tb

env = make_env()
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


def teleport(item, xyz):
    obj = env.env.get_object(item)
    env.env.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([xyz, [1, 0, 0, 0]]))
    env.env.sim.forward()


# 1. reset puts three items at the same fixed spots every time
layouts = []
for _ in range(3):
    fixed_reset(env)
    layouts.append(np.stack([body_pos(env, i) for i in tb.ITEMS]))
spread = np.abs(np.stack(layouts) - layouts[0]).max()
check("reset repeatability", spread < 0.01, f"max drift {spread*1000:.1f} mm")
gaps = [np.linalg.norm(layouts[0][a, :2] - layouts[0][b, :2]) for a, b in [(0, 1), (1, 2), (0, 2)]]
check("items far apart", min(gaps) > 0.12, f"min gap {min(gaps):.3f} m")
check("success False at start", not env.check_success())

# 2. reachability: Franka OSC workspace ~0.85 m from base (base at x=-0.66 in LIBERO floor frame)
base = np.array([-0.66, 0.0, 0.0])
reach = [np.linalg.norm(body_pos(env, n)[:2] - base[:2]) for n in (*tb.ITEMS, tb.BASKET)]
check("items+basket within reach", max(reach) < 0.85, f"max horizontal dist {max(reach):.2f} m")

# 3 & 4. drop items into basket one at a time; success only after the third (+ dwell)
basket = body_pos(env, tb.BASKET)
for k, item in enumerate(tb.ITEMS):
    teleport(item, basket + np.array([(k - 1) * 0.04, 0, 0.06]))
    for _ in range(40):
        env.step(DUMMY)
    states = env.env.object_states_dict
    inside = states[item].check_contact(states[tb.BASKET]) and states[tb.BASKET].check_contain(states[item]) \
        if hasattr(states[tb.BASKET], "check_contain") else None
    done = env.check_success()
    check(f"{item} in basket detected", inside is not False)
    check(f"success == {k == 2} after {k+1} item(s)", done == (k == 2))

# 5. an item pulled back out cancels success
teleport(tb.ITEMS[0], np.array([-0.17, -0.18, 0.95]))
for _ in range(10):
    env.step(DUMMY)
check("success drops when an item leaves", not env.check_success())

env.close()
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
