"""
plot_sequence.py
----------------
Load a trained model, give your own preference vectors [lambda_E, lambda_T],
and plot the task visiting sequence of the policy in 3D (same scenario for every vector).

Files needed in the same folder: train_ppo.py, test_ppo.py, scenario_gen.py, drone_energy.py
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import train_ppo as tp
import test_ppo as tpol                     # reuses load_model(), rollout(), draw_tour()
from scenario_gen import generate_scenario

# ======================================================================
# SETTINGS  (edit here only)
# ======================================================================
RUN_ID = "ts3-run_004"             # used for the output folder only
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "checkpoints",
    "ts3-run_004",
    "ts3-run_004_it05000_ep10000.pt",
)                                  # set this to the exact checkpoint to visualize
PREFERENCES = [                    # each row = [lambda_E, lambda_T]; rows are normalised to sum to 1
    [1.0, 0.0],                    # only energy
    [0.8, 0.2],                    # mostly energy
    [0.5, 0.5],                    # balanced
    [0.2, 0.8],                    # mostly makespan
    [0.3, 0.7],                    # mostly makespan
    [0.0, 1.0],                    # only makespan
    
    
]
SEED = 10001                       # scenario seed
N_TASKS = None                     # None = task count follows the robot category
GREEDY = True                      # True = argmax actions, False = sample
PREF_SWITCH_MAX_INJECTIONS = 2  # None = use checkpoint setting; set an int to override
PREF_SWITCH_TASK_THRESHOLD = 6  # None = use checkpoint setting; set an int to override
SAVE_FIG = True
SHOW_FIG = False
OUT_DIR = os.path.join("test_results", RUN_ID, "sequence")
MAX_COLS = 3                       # 3D plots per row


# ======================================================================
# Plot
# ======================================================================
def plot_sequences(pos, depot, orders, speed_fracs, ret_speed_fracs, prefs, E, T,
                   pref_switches, vmax, seed, cat):
    n = len(prefs)
    ncols = min(MAX_COLS, n)
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(6 * ncols, 5.5 * nrows + 0.8))
    for b in range(n):
        ax = fig.add_subplot(nrows, ncols, b + 1, projection="3d")
        tpol.draw_tour(
            ax, pos, depot, orders[b], speed_fracs[b], ret_speed_fracs[b], vmax,
            f"initial lambda = [{prefs[b][0]:.2f}, {prefs[b][1]:.2f}]\n"
            f"E = {E[b]:.0f} Wh   T = {T[b]:.0f} s",
            pref_switches[b])

    speed_map = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap="viridis")
    speed_map.set_array([])
    fig.colorbar(speed_map, ax=fig.axes, shrink=0.72, pad=0.08,
                 label="Flight speed / maximum speed")
    fig.legend(handles=[
        Line2D([0], [0], marker="x", color="red", lw=0, label="not served"),
        Line2D([0], [0], marker="*", markerfacecolor="red", markeredgecolor="black",
               lw=0, label="preference switch"),
    ], loc="lower center", ncol=2, fontsize=9)
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
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")
    tpol.CKPT_PATH = MODEL_PATH
    tpol.GREEDY = GREEDY
    policy = tpol.load_model()
    if PREF_SWITCH_MAX_INJECTIONS is not None:
        tp.hp.pref_switch_max_injections = PREF_SWITCH_MAX_INJECTIONS
    if PREF_SWITCH_TASK_THRESHOLD is not None:
        tp.hp.pref_switch_task_threshold = PREF_SWITCH_TASK_THRESHOLD
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
    tasks, speed_fracs, valid, ret_speed_fracs, pref_switches = tpol.rollout(
        policy, env, task_f, robot_f, lam)
    steps = tasks.shape[0]
    orders = [[int(tasks[t, b]) for t in range(steps) if valid[t, b]] for b in range(B)]
    episode_speeds = [[float(speed_fracs[t, b]) for t in range(steps) if valid[t, b]]
                      for b in range(B)]
    E, T = env.E_used.cpu().numpy(), env.T_used.cpu().numpy()
    unserved = (~env.visited).sum(1).cpu().numpy()

    # ---- 5. print + plot ---------------------------------------------------------------
    print(f"\nSeed {SEED}, category {cat}, {env.N} tasks")
    for b in range(B):
        print(f"[lam_E, lam_T] = [{prefs[b][0]:.2f}, {prefs[b][1]:.2f}]  "
              f"E = {E[b]:.1f} Wh  T = {T[b]:.0f} s  unserved = {int(unserved[b])}")
        print(f"    order: {orders[b]}")
        print(f"    outbound speed fractions: {[round(v, 3) for v in episode_speeds[b]]}")
        print(f"    return speed fraction: {ret_speed_fracs[b]:.3f}")
        for event in pref_switches[b]:
            print(f"    preference switch before step {event['step']} "
                  f"(task {event['task']}): {event['from']} -> {event['to']}")
    plot_sequences(pos, env.depot.cpu().numpy(), orders, episode_speeds, ret_speed_fracs,
                   prefs, E, T, pref_switches, env.vmax, SEED, cat)
    if SHOW_FIG:
        plt.show()


if __name__ == "__main__":
    main()
