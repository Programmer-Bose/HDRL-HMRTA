"""
infer_and_plot.py
------------------
End-to-end TEST / INFERENCE script (no training).

Set N_ROBOTS and N_TASKS below, run this file, and it will:
    1. generate one scenario (scenario_gen.generate_scenario)
    2. run the trained ASSIGNMENT layer (train_assignment.AssignmentPolicy)
       to decide which robot gets which task
    3. run the trained SEQUENCER layer (train_ppo.FiLMPointerPolicy, frozen)
       for every robot to decide visiting order + speeds
    4. plot the resulting fleet routes in 3D, and print a summary table

Both checkpoints are optional: if a path doesn't exist, that network falls
back to its random initialisation (clearly logged) so you can sanity-check
the full pipeline shape/wiring before you have trained weights.

Files needed in the same folder: scenario_gen.py, drone_energy.py,
train_ppo.py, train_assignment.py
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from scenario_gen import generate_scenario, DEPOTS, CATEGORIES
from train_assignment import (
    AssignmentPolicy, hp as assign_hp,
    fleet_task_features, all_robot_features, decode_assignment,
    w_to_lambda, fleet_task_features_batched, batched_policy_act,
    BatchedFleetEnv, load_frozen_sequencer,
)

# ======================================================================
# ---- SET THESE MANUALLY -----------------------------------------------
N_ROBOTS = 8
N_TASKS = 20
SEED = 123
PREFERENCE = (0.34, 0.33, 0.33)    # [w_makespan, w_energy, w_variance], must sum to ~1

ASSIGN_CHECKPOINT = "checkpoints_assign/assign-run_002/assign-run_002_it00020.pt"   # <-- set once trained
# ======================================================================

DEVICE = assign_hp.device


def load_assignment_policy():
    policy = AssignmentPolicy().to(DEVICE)
    if os.path.exists(ASSIGN_CHECKPOINT):
        ckpt = torch.load(ASSIGN_CHECKPOINT, map_location=DEVICE, weights_only=True)
        policy.load_state_dict(ckpt["model"])
        print(f"Loaded assignment policy from {ASSIGN_CHECKPOINT}")
    else:
        print(f"[WARNING] assignment checkpoint not found at '{ASSIGN_CHECKPOINT}'. "
              f"Using a randomly-initialised assignment network (pipeline sanity-check only).")
    policy.eval()
    return policy


def run_once(n_robots, n_tasks, seed, w_tuple):
    robots, task_pos, task_payload, task_service, _, err = generate_scenario(
        n_robots, n_tasks, seed=seed, return_meta=True)
    if err is not None:
        raise RuntimeError(f"scenario generation failed: {err}")

    assign_policy = load_assignment_policy()
    seq_policy = load_frozen_sequencer()
    w = torch.tensor(w_tuple, dtype=torch.float32, device=DEVICE)

    # ---- 1. assignment layer (one-shot forward + forced decode) ----------
    fleet_cap_ref = max(CATEGORIES["payload_kg"])
    task_f = fleet_task_features(task_pos, task_payload, task_service, fleet_cap_ref)
    robot_f = all_robot_features(robots)
    with torch.no_grad():
        logits, _ = assign_policy(task_f, robot_f, w)
        assign, _, _ = decode_assignment(logits, deterministic=True)
    assign_np = assign.cpu().numpy()

    # ---- 2. sequencer layer, batched over robots, WITH route recording ---
    env = BatchedFleetEnv(robots, assign_np, task_pos, task_payload, task_service, DEVICE)
    env.reset()
    M = env.M
    lam = w_to_lambda(w)[None, :].expand(M, 2)
    seq_task_f = fleet_task_features_batched(env.pos, env.pay, env.srv, env.cap)
    pad_mask = ~env.valid

    paths = [[env.depot[r].cpu().numpy()] for r in range(M)]   # each robot's route, starts at its depot
    with torch.no_grad():
        for _ in range(env.Nmax):
            if not env.active.any():
                break
            active_before = env.active.clone()
            state = env.observe()
            task, speed_frac, x_ret = batched_policy_act(
                seq_policy, seq_task_f, robot_f, state, lam, env.ok, env.speed_feat,
                env.load / env.cap, env.T_used / 3600.0, env.depot, pad_mask)
            next_pos = env.pos[torch.arange(M, device=DEVICE), task].cpu().numpy()
            for r in range(M):
                if active_before[r]:
                    paths[r].append(next_pos[r])
            env.step(task, speed_frac, x_ret)
            newly_finished = active_before & ~env.active
            for r in range(M):
                if newly_finished[r]:
                    paths[r].append(env.depot[r].cpu().numpy())   # return leg, for the plot

    load_frac = (env.pay * env.valid).sum(1) / env.cap.clamp(min=1e-6)
    results = {
        "robots": robots, "task_pos": task_pos, "task_payload": task_payload,
        "assign": assign_np, "paths": paths,
        "E_used": env.E_used.cpu().numpy(), "T_used": env.T_used.cpu().numpy(),
        "soc_drop": (env.E_used / env.usable.clamp(min=1e-6)).cpu().numpy() * 100.0,
        "load_frac": load_frac.cpu().numpy(),
        "makespan": env.T_used.max().item(),
    }
    return results


def print_summary(res):
    robots = res["robots"]
    print("\n=== Per-robot summary ===")
    print(f"{'R':>3} {'cat':>3} {'#tasks':>7} {'load%':>7} {'SOC drop%':>10} {'T (s)':>8}")
    for r in range(len(robots["category"])):
        n_tasks_r = int((res["assign"] == r).sum())
        print(f"{r:>3} {robots['category'][r]:>3} {n_tasks_r:>7} "
              f"{res['load_frac'][r]*100:>6.1f}% {res['soc_drop'][r]:>9.1f}% {res['T_used'][r]:>8.0f}")
    print(f"\nMakespan: {res['makespan']:.0f} s "
          f"({res['makespan']/60:.1f} min)")


def plot_result(res, out_path="/mnt/user-data/outputs/fleet_routes.png"):
    robots, task_pos, task_payload = res["robots"], res["task_pos"], res["task_payload"]
    M = len(robots["category"])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")
    colors = [cmap(r % 20) for r in range(M)]

    ax.scatter(*DEPOTS.T, c="black", marker="s", s=100, label="Depots")
    for i, (p, w) in enumerate(zip(task_pos, task_payload)):
        r = res["assign"][i]
        ax.scatter(*p, c=[colors[r]], marker="o", s=40)
        ax.text(p[0], p[1], p[2], f" T{i}\n{w:.1f}kg", fontsize=6)

    for r in range(M):
        path = np.array(res["paths"][r])
        if len(path) > 1:
            ax.plot(path[:, 0], path[:, 1], path[:, 2], c=colors[r], linewidth=1.8,
                    label=f"R{r} (cat {robots['category'][r]})")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"Fleet assignment + routes ({M} robots, {len(task_pos)} tasks)\n"
                 f"makespan={res['makespan']:.0f}s")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    res = run_once(N_ROBOTS, N_TASKS, SEED, PREFERENCE)
    print_summary(res)
    plot_result(res)
