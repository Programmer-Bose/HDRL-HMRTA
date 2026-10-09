"""
plot_sequence.py
----------------
Load a trained model, give your own preference vectors [lambda_E, lambda_T],
and plot the task visiting sequence of the policy in 3D (same scenario for every vector).

Files needed in the same folder: train_ppo.py, test_policy.py, scenario_gen.py, drone_energy.py
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import train_ppo as tp
import test_ppo as tpol                     # reuses load_model(), rollout(), draw_tour()
from scenario_gen import generate_scenario
from drone_energy import SPEED_LEVELS

# ======================================================================
# SETTINGS  (edit here only)
# ======================================================================
MODEL_PATH = "checkpoints/run_004/run_004_it01000_ep05000.pt"
PREFERENCES = [                    # each row = [lambda_E, lambda_T]; rows are normalised to sum to 1
    [1.0, 0.0],                    # only energy
    [0.8, 0.2],                    # mostly energy
    [0.5, 0.5],                    # balanced
    [0.2, 0.8],                    # mostly makespan
    [0.3, 0.7],                    # mostly makespan
    [0.0, 1.0],                    # only makespan
    
    
]
SEED = 2000                       # scenario seed
N_TASKS = None                     # None = task count follows the robot category
GREEDY = True                      # True = argmax actions, False = sample
SAVE_FIG = False
SHOW_FIG = True
OUT_DIR = os.path.join("test_results", "sequence")
MAX_COLS = 3                       # 3D plots per row


# ======================================================================
# Plot
# ======================================================================
def plot_sequences(pos, depot, orders, levels, ret_level, prefs, E, T, seed, cat):
    n = len(prefs)
    ncols = min(MAX_COLS, n)
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(6 * ncols, 5.5 * nrows + 0.8))
    for b in range(n):
        ax = fig.add_subplot(nrows, ncols, b + 1, projection="3d")
        tpol.draw_tour(ax, pos, depot, orders[b], levels[b], ret_level,
                       f"[lam_E, lam_T] = [{prefs[b][0]:.2f}, {prefs[b][1]:.2f}]\n"
                       f"E = {E[b]:.0f} Wh   T = {T[b]:.0f} s")

    S = len(SPEED_LEVELS)
    handles = [Line2D([0], [0], color=plt.cm.viridis(i / max(S - 1, 1)), lw=3,
                      label=f"{f:.2f} x vmax") for i, f in enumerate(SPEED_LEVELS)]
    handles += [Line2D([0], [0], marker="x", color="red", lw=0, label="not served")]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), fontsize=9)
    fig.suptitle(f"Task visiting sequence (numbers) - seed {seed}, category {cat}")
    if SAVE_FIG:
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, f"sequence_seed{seed}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")


# ======================================================================
# Main
# ======================================================================
def main():
    # ---- 1. model ------------------------------------------------------------
    tpol.CKPT_PATH = MODEL_PATH
    tpol.GREEDY = GREEDY
    policy = tpol.load_model()
    dev = tp.hp.device

    # ---- 2. preference vectors ----------------------------------------------------
    prefs = np.array(PREFERENCES, dtype=float)
    assert prefs.ndim == 2 and prefs.shape[1] == 2, "each preference must be [lambda_E, lambda_T]"
    assert (prefs >= 0).all() and (prefs.sum(1) > 0).all(), "values must be >= 0 and not all zero"
    prefs = prefs / prefs.sum(1, keepdims=True)
    lam = torch.as_tensor(prefs, dtype=torch.float32, device=dev)                # [B,2]
    B = len(prefs)

    # ---- 3. scenario + environment (same scenario for every preference) ---------
    robots, pos, pay, srv, _, err = generate_scenario(1, N_TASKS, seed=SEED, return_meta=True)
    if err is not None:
        raise SystemExit(f"Scenario error: {err}")
    env = tp.SingleDroneEnv(robots, pos, pay, srv, B)
    env.reset()
    if not env.ok.any():
        raise SystemExit("No task fits the battery for this scenario.")
    task_f = tp.task_features(pos, pay, srv, env.cap)
    robot_f = tp.robot_features(robots)
    cat = int(robots["category"][0])

    # ---- 4. policy rollout -----------------------------------------------------------
    tasks, levels, valid = tpol.rollout(policy, env, task_f, robot_f, lam)
    steps = tasks.shape[0]
    orders = [[int(tasks[t, b]) for t in range(steps) if valid[t, b]] for b in range(B)]
    lvls = [[int(levels[t, b]) for t in range(steps) if valid[t, b]] for b in range(B)]
    E, T = env.E_used.cpu().numpy(), env.T_used.cpu().numpy()
    unserved = (~env.visited).sum(1).cpu().numpy()

    # ---- 5. print + plot ---------------------------------------------------------------
    print(f"\nSeed {SEED}, category {cat}, {env.N} tasks")
    for b in range(B):
        print(f"[lam_E, lam_T] = [{prefs[b][0]:.2f}, {prefs[b][1]:.2f}]  "
              f"E = {E[b]:.1f} Wh  T = {T[b]:.0f} s  unserved = {int(unserved[b])}")
        print(f"    order: {orders[b]}")
        print(f"    speed levels: {lvls[b]}")
    plot_sequences(pos, env.depot.cpu().numpy(), orders, lvls, tp.hp.return_speed_level,
                   prefs, E, T, SEED, cat)
    if SHOW_FIG:
        plt.show()


if __name__ == "__main__":
    main()
