"""
ta_infer_and_plot.py
--------------------
End-to-end TEST / INFERENCE script for the top-layer policy in ta_train.py.

Set SEED and PREFERENCES below, then run this file. It generates an inventory
and task set with ta_train.sample_scenario, evaluates the same scenario for each
preference vector, and plots each assignment with the training sequencer's fixed
battery-feasibility preference.

The reported objectives and reward match ta_train. Tasks that cannot fit within
any remaining robot capacity are left unassigned, matching
ta_train.decode_assignment. Missing checkpoints fall back to random
initialisation for pipeline sanity checks only.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from scenario_gen import (
    CATEGORIES,
    DEPOTS,
    generate_fleet_inventory,
    generate_fleet_tasks,
)
from ta_train import (
    AssignmentPolicy,
    BatchedFleetEnv,
    SEQ_FIXED_LAM,
    all_robot_features,
    batched_policy_act,
    decode_assignment,
    fleet_task_features,
    fleet_task_features_batched,
    fleet_objectives,
    hp as assign_hp,
    load_frozen_sequencer,
    scalarize,
)


# ---- Set these for a test run ------------------------------------------
N_ROBOTS = 15
N_TASKS = 30
SEED = 42
PREFERENCES = [
    (0.8, 0.1, 0.1),
    (0.1, 0.8, 0.1),
    (0.1, 0.1, 0.8),
]  # each: [w_launch_cost, w_peak_load, w_depot_reserve]

ASSIGN_CHECKPOINT = os.path.join(
    assign_hp.save_root,
    assign_hp.run_id,
    "assign-run_001_it00025.pt",
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


def run_preference(
    robots,
    task_pos,
    task_payload,
    task_service,
    seed,
    w_tuple,
    assign_policy,
    seq_policy,
):
    fleet_cap_ref = max(CATEGORIES["payload_kg"])
    task_f = fleet_task_features(task_pos, task_payload, task_service, fleet_cap_ref)
    robot_f = all_robot_features(robots)
    capacity = torch.as_tensor(
        robots["payload_kg"], dtype=torch.float32, device=DEVICE
    )
    payload = torch.as_tensor(task_payload, dtype=torch.float32, device=DEVICE)
    w = torch.tensor(w_tuple, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits, _ = assign_policy(task_f, robot_f, w)
        # ta_train's decoder masks each robot once its payload capacity is used;
        # its final logit column represents a task left unassigned.
        assign, _, _ = decode_assignment(
            logits, payload, capacity, deterministic=True
        )
    assign_np = assign.cpu().numpy()

    f1n, f2, f3, launched = fleet_objectives(robots, assign_np, task_payload)
    n_unassigned = int((assign_np == len(robots["category"])).sum())
    env = BatchedFleetEnv(
        robots, assign_np, task_pos, task_payload, task_service, DEVICE
    )
    env.reset()
    n_fleet_robots = env.M
    lam = torch.as_tensor(SEQ_FIXED_LAM, dtype=torch.float32, device=DEVICE)[
        None, :
    ].expand(n_fleet_robots, 2)
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

    unserved = (~env.visited) & env.valid
    n_battery_violations = int(unserved.sum().item())
    reward = scalarize(f1n, f2, f3, n_unassigned, n_battery_violations, w)
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
        "n_battery_violations": n_battery_violations,
        "launched": launched,
        "paths": paths,
        "f1n_launch_cost": f1n,
        "f2_peak_load": f2,
        "f3_reserve": f3,
        "reward": reward,
    }


def run_preferences(n_robots, n_tasks, seed, preferences):
    if n_robots <= 0:
        raise ValueError("n_robots must be greater than zero.")
    if n_tasks <= 0:
        raise ValueError("n_tasks must be greater than zero.")

    inventory_seed, selection_seed, task_seed = np.random.SeedSequence(seed).spawn(3)
    robots = generate_fleet_inventory(
        seed=inventory_seed,
        count_range=assign_hp.robots_per_cat_depot_range,
    )
    if len(robots["category"]) < n_robots:
        n_depots = len(DEPOTS)
        n_categories = len(CATEGORIES["payload_kg"])
        robots_per_category = max(
            assign_hp.robots_per_cat_depot_range[0],
            int(np.ceil(n_robots / (n_depots * n_categories))),
        )
        robots = generate_fleet_inventory(
            seed=inventory_seed,
            count_range=(robots_per_category, robots_per_category),
        )

    if n_robots < len(robots["category"]):
        rng = np.random.default_rng(selection_seed)
        selected = np.sort(
            rng.choice(len(robots["category"]), size=n_robots, replace=False)
        )
        robots = {key: values[selected] for key, values in robots.items()}

    task_pos, task_payload, task_service, err = generate_fleet_tasks(
        robots, n_tasks, seed=task_seed
    )
    if err is not None:
        raise RuntimeError(f"scenario generation failed: {err}")

    assign_policy = load_assignment_policy()
    seq_policy = load_frozen_sequencer()
    results = [
        run_preference(
            robots,
            task_pos,
            task_payload,
            task_service,
            seed,
            preference,
            assign_policy,
            seq_policy,
        )
        for preference in preferences
    ]
    if not results:
        return results

    for result in results:
        result["n_robots"] = n_robots
        result["n_tasks"] = n_tasks

    diversity_bonus = 0.0
    for i, left in enumerate(results):
        for right in results[i + 1:]:
            preference_distance = sum(
                abs(a - b)
                for a, b in zip(left["preference"], right["preference"])
            )
            same_assignment_fraction = float(
                np.mean(left["assign"] == right["assign"])
            )
            diversity_bonus -= (
                assign_hp.diversity_coef
                * preference_distance
                * same_assignment_fraction
            )

    if results:
        for result in results:
            result["diversity_bonus"] = diversity_bonus / len(results)
            result["reward"] += result["diversity_bonus"]
    return results


def print_scenario(res):
    print("=== Input scenario ===")
    print(f"Seed: {res['seed']}")
    print(f"Robots: {res['n_robots']}")
    print(f"Tasks: {res['n_tasks']}")
    print(f"Total robot payload capacity: {res['total_robot_capacity']:.2f} kg")
    print(f"Total task payload: {res['total_task_capacity']:.2f} kg")
    print(
        "Task payload / fleet capacity: "
        f"{res['total_task_capacity'] / res['total_robot_capacity']:.1%}"
    )


def print_summary(res):
    print(f"\n=== Preference: {res['preference']} ===")
    assignment_vector = [
        int(robot) if int(robot) < res["n_robots"] else None
        for robot in res["assign"]
    ]
    print(f"Assignment vector (task index -> robot index): {assignment_vector}")
    print("(None indicates an unassigned task.)")
    print(f"Unassigned task count: {res['n_unassigned']}")
    print(f"Battery-feasibility violations: {res['n_battery_violations']}")
    print("Objective values (matching ta_train):")
    print(f"  f1 normalized launch cost: {res['f1n_launch_cost']:.6f} (minimize)")
    print(
        f"  f2 peak launched-robot load ratio: "
        f"{res['f2_peak_load']:.6f} (minimize)"
    )
    print(
        f"  f3 minimum depot reserve readiness: "
        f"{res['f3_reserve']:.6f} (maximize)"
    )
    print(f"Training-style reward: {res['reward']:.6f}")
    if res.get("diversity_bonus", 0.0):
        print(f"  (includes diversity bonus {res['diversity_bonus']:.6f})")


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
        f"ta_train assignment + battery check ({n_fleet_robots} robots, "
        f"{len(task_pos)} tasks)\n"
        f"launch cost={res['f1n_launch_cost']:.3f}, "
        f"peak load={res['f2_peak_load']:.3f}, reserve={res['f3_reserve']:.3f}"
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
    results = run_preferences(N_ROBOTS, N_TASKS, SEED, PREFERENCES)
    if results:
        print_scenario(results[0])
    for result in results:
        print_summary(result)
        plot_result(result)
