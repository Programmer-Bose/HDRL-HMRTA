"""
ta_infer_and_plot.py
--------------------
End-to-end TEST / INFERENCE script for the top-layer policy in ta_train.py.

Set N_ROBOTS and N_TASKS below, then run this file. It generates one scenario,
uses the trained assignment policy with its payload-capacity mask, runs the
frozen sequencer for each robot, and plots the fleet routes.

Tasks that cannot fit within any remaining robot capacity are left unassigned,
matching ta_train.decode_assignment. Missing checkpoints fall back to random
initialisation for pipeline sanity checks only.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from scenario_gen import CATEGORIES, DEPOTS, generate_scenario
from ta_train import (
    AssignmentPolicy,
    BatchedFleetEnv,
    all_robot_features,
    batched_policy_act,
    decode_assignment,
    fleet_task_features,
    fleet_task_features_batched,
    hp as assign_hp,
    load_frozen_sequencer,
    w_to_lambda,
)


# ---- Set these for a test run ------------------------------------------
N_ROBOTS = 10
N_TASKS = 30
SEED = 42
PREFERENCE = (0.8, 0.1, 0.1)  # [w_makespan, w_energy, w_variance]

ASSIGN_CHECKPOINT = os.path.join(
    assign_hp.save_root,
    assign_hp.run_id,
    f"{assign_hp.run_id}_it{assign_hp.num_iterations:05d}.pt",
)
# ------------------------------------------------------------------------

DEVICE = assign_hp.device


def load_assignment_policy():
    policy = AssignmentPolicy().to(DEVICE)
    if os.path.exists(ASSIGN_CHECKPOINT):
        ckpt = torch.load(ASSIGN_CHECKPOINT, map_location=DEVICE, weights_only=True)
        policy.load_state_dict(ckpt["model"])
        print(f"Loaded assignment policy from {ASSIGN_CHECKPOINT}")
    else:
        print(
            f"[WARNING] assignment checkpoint not found at "
            f"'{ASSIGN_CHECKPOINT}'. Using a randomly-initialised assignment "
            "network (pipeline sanity-check only)."
        )
    policy.eval()
    return policy


def run_once(n_robots, n_tasks, seed, w_tuple):
    robots, task_pos, task_payload, task_service, _, err = generate_scenario(
        n_robots, n_tasks, seed=seed, return_meta=True
    )
    if err is not None:
        raise RuntimeError(f"scenario generation failed: {err}")

    assign_policy = load_assignment_policy()
    seq_policy = load_frozen_sequencer()
    w = torch.tensor(w_tuple, dtype=torch.float32, device=DEVICE)

    fleet_cap_ref = max(CATEGORIES["payload_kg"])
    task_f = fleet_task_features(task_pos, task_payload, task_service, fleet_cap_ref)
    robot_f = all_robot_features(robots)
    capacity = torch.as_tensor(
        robots["payload_kg"], dtype=torch.float32, device=DEVICE
    )
    payload = torch.as_tensor(task_payload, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits, _ = assign_policy(task_f, robot_f, w)
        # ta_train's decoder masks each robot once its payload capacity is used;
        # its final logit column represents a task left unassigned.
        assign, _, _ = decode_assignment(
            logits, payload, capacity, deterministic=True
        )
    assign_np = assign.cpu().numpy()

    env = BatchedFleetEnv(
        robots, assign_np, task_pos, task_payload, task_service, DEVICE
    )
    env.reset()
    n_fleet_robots = env.M
    lam = w_to_lambda(w)[None, :].expand(n_fleet_robots, 2)
    seq_task_f = fleet_task_features_batched(env.pos, env.pay, env.srv, env.cap)
    pad_mask = ~env.valid

    paths = [
        [env.depot[r].cpu().numpy()] for r in range(n_fleet_robots)
    ]
    with torch.no_grad():
        for _ in range(env.Nmax):
            if not env.active.any():
                break
            active_before = env.active.clone()
            state = env.observe()
            task, speed_frac, x_ret = batched_policy_act(
                seq_policy,
                seq_task_f,
                robot_f,
                state,
                lam,
                env.ok,
                env.speed_feat,
                env.load / env.cap,
                env.T_used / 3600.0,
                env.depot,
                pad_mask,
            )
            next_pos = env.pos[torch.arange(n_fleet_robots, device=DEVICE), task]
            next_pos = next_pos.cpu().numpy()
            for r in range(n_fleet_robots):
                if active_before[r]:
                    paths[r].append(next_pos[r])
            env.step(task, speed_frac, x_ret)
            newly_finished = active_before & ~env.active
            for r in range(n_fleet_robots):
                if newly_finished[r]:
                    paths[r].append(env.depot[r].cpu().numpy())

    load_frac = (env.pay * env.valid).sum(1) / env.cap.clamp(min=1e-6)
    mean_soc_drop = (
        env.E_used / env.usable.clamp(min=1e-6)
    ).mean().item()
    payload_var = load_frac.var(unbiased=False).item()
    n_unassigned = int((assign == n_fleet_robots).sum().item())
    return {
        "robots": robots,
        "task_pos": task_pos,
        "task_payload": task_payload,
        "n_robots": n_fleet_robots,
        "n_tasks": len(task_payload),
        "seed": seed,
        "preference": tuple(float(weight) for weight in w_tuple),
        "total_robot_capacity": float(np.sum(robots["payload_kg"])),
        "total_task_capacity": float(np.sum(task_payload)),
        "assign": assign_np,
        "n_unassigned": n_unassigned,
        "paths": paths,
        "E_used": env.E_used.cpu().numpy(),
        "T_used": env.T_used.cpu().numpy(),
        "soc_drop": (
            env.E_used / env.usable.clamp(min=1e-6)
        ).cpu().numpy() * 100.0,
        "load_frac": load_frac.cpu().numpy(),
        "makespan": env.T_used.max().item(),
        "mean_soc_drop": mean_soc_drop,
        "payload_var": payload_var,
    }


def print_summary(res):
    robots = res["robots"]
    print("=== Input scenario ===")
    print(f"Robots: {res['n_robots']}")
    print(f"Tasks: {res['n_tasks']}")
    print(f"Total robot payload capacity: {res['total_robot_capacity']:.2f} kg")
    print(f"Total task payload: {res['total_task_capacity']:.2f} kg")

    print("\n=== Per-robot summary ===")
    print(f"{'R':>3} {'cat':>3} {'#tasks':>7} {'load%':>7} {'SOC drop%':>10} {'T (s)':>8}")
    for r in range(len(robots["category"])):
        n_tasks_r = int((res["assign"] == r).sum())
        print(
            f"{r:>3} {robots['category'][r]:>3} {n_tasks_r:>7} "
            f"{res['load_frac'][r] * 100:>6.1f}% "
            f"{res['soc_drop'][r]:>9.1f}% {res['T_used'][r]:>8.0f}"
        )
    print(f"\nUnassigned tasks: {res['n_unassigned']}")
    print("\n=== Objective values (ta_train) ===")
    print(f"Makespan: {res['makespan']:.2f} s ({res['makespan'] / 60:.2f} min)")
    print(
        f"Mean SOC drop: {res['mean_soc_drop']:.6f} "
        f"({res['mean_soc_drop'] * 100:.2f}%)"
    )
    print(f"Payload utilization variance: {res['payload_var']:.6f}")


def plot_result(res, out_path=None):
    robots, task_pos, task_payload = (
        res["robots"],
        res["task_pos"],
        res["task_payload"],
    )
    n_fleet_robots = len(robots["category"])
    if out_path is None:
        preference = "-".join(f"{weight:.2f}" for weight in res["preference"])
        filename = (
            f"ta_routes_pref-{preference}_seed-{res['seed']}"
            f"_robots-{res['n_robots']}_tasks-{res['n_tasks']}.png"
        )
        out_path = os.path.join("outputs", filename)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")
    colors = [cmap(r % 20) for r in range(n_fleet_robots)]

    ax.scatter(*DEPOTS.T, c="black", marker="s", s=100, label="Depots")
    for i, (pos, payload) in enumerate(zip(task_pos, task_payload)):
        robot = res["assign"][i]
        if robot == n_fleet_robots:
            ax.scatter(*pos, c="gray", marker="x", s=50)
            ax.text(pos[0], pos[1], pos[2], f" T{i} unassigned", fontsize=6)
        else:
            ax.scatter(*pos, c=[colors[robot]], marker="o", s=40)
            ax.text(pos[0], pos[1], pos[2], f" T{i}\n{payload:.1f}kg", fontsize=6)

    for r in range(n_fleet_robots):
        path = np.asarray(res["paths"][r])
        if len(path) > 1:
            ax.plot(
                path[:, 0],
                path[:, 1],
                path[:, 2],
                c=colors[r],
                linewidth=1.8,
                label=f"R{r} (cat {robots['category'][r]})",
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        f"ta_train assignment + routes ({n_fleet_robots} robots, "
        f"{len(task_pos)} tasks)\n"
        f"unassigned={res['n_unassigned']}, makespan={res['makespan']:.0f}s"
    )
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    plt.tight_layout()
    output_dir = os.path.dirname(out_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    result = run_once(N_ROBOTS, N_TASKS, SEED, PREFERENCE)
    print_summary(result)
    plot_result(result)
