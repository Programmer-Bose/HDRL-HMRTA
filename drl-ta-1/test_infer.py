"""
test_infer.py
-------------
Load a trained assignment policy + the frozen sequencer, build ONE fleet
(same number of robots per category at every depot), and run it under a
LIST of preference vectors w = [w1_cost, w2_peak_load, w3_reserve]. For
each w: decode the assignment (deterministic), evaluate its objectives,
print them, and save a 3D plot of the result.

Edit the CONFIG block below, then run:  python test_infer.py
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from scenario_gen import generate_fleet_tasks, DEPOTS, CATEGORIES, LAUNCH_COST
from train_ppo import FiLMPointerPolicy
from ta_train import (
    AssignmentPolicy, decode_assignment, evaluate_assignment,
    fleet_task_features, all_robot_features, hp as ta_hp,
    BatchedFleetEnv, fleet_task_features_batched, batched_policy_act, SEQ_FIXED_LAM,
)

# ======================================================================
# CONFIG -- edit these
# ======================================================================
ROBOTS_PER_CATEGORY = [2, 1, 1, 1]   # one count per category (0..3), SAME at every depot
N_TASKS = 75
SEED = 42

PREFERENCES = [                  # each is [w1_launch_cost, w2_peak_load, w3_reserve]
    [1.00, 0.00, 0.00],
    [0.00, 1.00, 0.00],
    [0.00, 0.00, 1.00],
    [0.34, 0.33, 0.33],
]

ASSIGN_CHECKPOINT = "checkpoints_assign/assign-run_001/assign-run_001_it00075.pt"
SEQ_CHECKPOINT = ta_hp.lower_checkpoint     # reuses the path set in ta_train.HP
OUT_DIR = "test_outputs"
DEVICE = ta_hp.device


# ======================================================================
# Checkpoint loading
# ======================================================================
def load_assignment_policy(path, device):
    policy = AssignmentPolicy().to(device)
    loaded = False
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        policy.load_state_dict(ckpt["model"])
        print(f"Loaded assignment policy from {path}")
        loaded = True
    else:
        print("=" * 70)
        print(f"[WARNING] assignment checkpoint NOT FOUND at '{path}'.")
        print("Running with a RANDOMLY-INITIALISED policy -- results below will")
        print("NOT vary meaningfully with w, and objective values are meaningless.")
        print("Set ASSIGN_CHECKPOINT to a real .pt file from checkpoints_assign/.")
        print("=" * 70)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy, loaded


def load_sequencer(path, device):
    policy = FiLMPointerPolicy().to(device)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        policy.load_state_dict(ckpt["model"])
        print(f"Loaded sequencer from {path}")
    else:
        print(f"[WARNING] sequencer checkpoint not found at '{path}'. "
              f"Running with a randomly-initialised sequencer.")
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


# ======================================================================
# Scenario: EXACT robots-per-category counts, replicated identically at
# every depot (e.g. ROBOTS_PER_CATEGORY = [5, 3, 2, 4] -> every depot gets
# 5 cat-0, 3 cat-1, 2 cat-2, 4 cat-3 robots -- no randomness in the counts).
# ======================================================================
def build_fleet(counts_per_category, n_depots=None):
    n_categories = len(CATEGORIES["payload_kg"])
    if len(counts_per_category) != n_categories:
        raise ValueError(f"ROBOTS_PER_CATEGORY must have {n_categories} entries "
                          f"(one per category), got {len(counts_per_category)}.")
    if n_depots is None:
        n_depots = len(DEPOTS)

    category_list, depot_list = [], []
    for d in range(n_depots):
        for c, n in enumerate(counts_per_category):
            category_list += [c] * int(n)
            depot_list += [d] * int(n)

    categories = np.array(category_list, dtype=int)
    depot_id = np.array(depot_list, dtype=int)

    return {
        "category": categories,
        "payload_kg": np.array(CATEGORIES["payload_kg"])[categories],
        "flight_min": np.array(CATEGORIES["flight_min"])[categories],
        "battery_wh": np.array(CATEGORIES["battery_wh"])[categories],
        "speed_ms": np.array(CATEGORIES["speed_ms"])[categories],
        "depot_id": depot_id,
        "launch_cost": np.array(LAUNCH_COST)[categories],
    }


def build_scenario(seed):
    robots = build_fleet(ROBOTS_PER_CATEGORY)
    task_pos, task_payload, task_service, err = generate_fleet_tasks(
        robots, N_TASKS, seed=seed + 1)
    if err is not None:
        raise RuntimeError(f"Task generation failed: {err}")
    return robots, task_pos, task_payload, task_service


# ======================================================================
# Route extraction: re-run the SAME sequencer rollout evaluate_assignment
# does (fixed SEQ_FIXED_LAM), but keep each robot's position history so we
# can actually draw the flight path, not just the final task assignment.
# ======================================================================
@torch.no_grad()
def get_fleet_routes(seq_policy, robots, task_pos, task_payload, task_service, assign):
    env = BatchedFleetEnv(robots, assign, task_pos, task_payload, task_service, DEVICE)
    env.reset()
    lam = torch.as_tensor(SEQ_FIXED_LAM, dtype=torch.float32, device=DEVICE)[None, :].expand(env.M, 2)
    task_f = fleet_task_features_batched(env.pos, env.pay, env.srv, env.cap)
    robot_f = all_robot_features(robots)
    pad_mask = ~env.valid

    depot_np = env.depot.cpu().numpy()
    routes = [[depot_np[r].copy()] for r in range(env.M)]   # every route starts at its depot
    launched = env.valid.any(dim=1).cpu().numpy()

    for _ in range(env.Nmax):
        if not env.active.any():
            break
        was_active = env.active.clone()
        state = env.observe()
        task, speed_frac, x_ret = batched_policy_act(
            seq_policy, task_f, robot_f, state, lam, env.ok, env.speed_feat,
            env.load / env.cap, env.T_used / 3600.0, env.depot, pad_mask)
        env.step(task, speed_frac, x_ret)
        cur = env.cur.cpu().numpy()
        for r in range(env.M):
            if was_active[r]:
                routes[r].append(cur[r].copy())

    for r in range(env.M):
        if launched[r]:
            routes[r].append(depot_np[r].copy())        # fly home at the end
    return routes, launched


# ======================================================================
# Plotting: depots as squares, robots as triangles (gray = idle), tasks as
# dots colored by the robot they were assigned to (black = unassigned),
# flight paths drawn as lines colored to match their robot
# ======================================================================
def plot_assignment(robots, task_pos, assign, w, routes, launched, out_path):
    M = len(robots["category"])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")

    ax.scatter(*DEPOTS.T, c="purple", marker="s", s=140, label="Depots")
    for d, pos in enumerate(DEPOTS):
        ax.text(pos[0], pos[1], pos[2] - 0.03, f"D{d}", fontsize=8, color="purple")

    # robots: jittered around their depot so same-depot robots don't overlap
    robot_xyz = DEPOTS[robots["depot_id"]] + 0.015 * (np.arange(M)[:, None] % 5)
    for r in range(M):
        color = cmap(r % 20) if launched[r] else "lightgray"
        marker = "^" if launched[r] else "v"
        ax.scatter(*robot_xyz[r], c=[color], marker=marker, s=60)

    for j in range(len(task_pos)):
        r = int(assign[j])
        color = cmap(r % 20) if r < M else "black"
        ax.scatter(*task_pos[j], c=[color], marker="o", s=25)

    # flight paths: depot -> task -> task -> ... -> depot, per launched robot
    for r in range(M):
        if not launched[r] or len(routes[r]) < 2:
            continue
        pts = np.array(routes[r])
        ax.plot3D(pts[:, 0], pts[:, 1], pts[:, 2], c=cmap(r % 20), linewidth=1.2, alpha=0.7)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Assignment for w={np.round(w, 2).tolist()}  "
                 f"(launched {int(launched.sum())}/{M} robots)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"   saved {out_path}")


# ======================================================================
# Main
# ======================================================================
@torch.no_grad()
def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    assign_policy, assign_loaded = load_assignment_policy(ASSIGN_CHECKPOINT, DEVICE)
    seq_policy = load_sequencer(SEQ_CHECKPOINT, DEVICE)

    robots, task_pos, task_payload, task_service = build_scenario(SEED)
    fleet_cap_ref = max(CATEGORIES["payload_kg"])
    task_f = fleet_task_features(task_pos, task_payload, task_service, fleet_cap_ref)
    robot_f = all_robot_features(robots)
    capacity = torch.as_tensor(robots["payload_kg"], dtype=torch.float32, device=DEVICE)
    payload_t = torch.as_tensor(task_payload, dtype=torch.float32, device=DEVICE)

    print(f"Fleet: {len(robots['category'])} robots across {len(DEPOTS)} depots "
          f"({ROBOTS_PER_CATEGORY} per category, same at every depot) | "
          f"{len(task_payload)} tasks\n")

    for w_list in PREFERENCES:
        w = torch.as_tensor(w_list, dtype=torch.float32, device=DEVICE)
        logits, _ = assign_policy(task_f, robot_f, w)
        assign, _, _ = decode_assignment(logits, payload_t, capacity, deterministic=True)
        assign_np = assign.cpu().numpy()

        f1n, f2, f3, n_un, n_bat = evaluate_assignment(
            seq_policy, robots, task_pos, task_payload, task_service, assign_np)

        print(f"w={w_list} -> f1n(cost)={f1n:.3f}  f2(peak_load)={f2:.3f}  "
              f"f3(reserve)={f3:.3f}  unassigned={n_un}  battery_violations={n_bat}")

        routes, launched = get_fleet_routes(
            seq_policy, robots, task_pos, task_payload, task_service, assign_np)

        fname = "assign_w" + "_".join(f"{x:.2f}" for x in w_list) + ".png"
        plot_assignment(robots, task_pos, assign_np, np.array(w_list), routes, launched,
                         os.path.join(OUT_DIR, fname))

    if not assign_loaded:
        print("\n[REMINDER] the assignment policy above was randomly initialised -- "
              "train it first, then point ASSIGN_CHECKPOINT at a saved checkpoint.")
    print(f"All plots saved under '{OUT_DIR}/'.")


if __name__ == "__main__":
    run()