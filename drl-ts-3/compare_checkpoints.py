"""
compare_checkpoints.py
----------------------
Evaluate every saved checkpoint for one run ID on the same test scenarios,
then save per-scenario and aggregate comparison plots plus a CSV.
"""

import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch

import test_ppo as evaluator
import train_ppo as tp
from scenario_gen import CATEGORIES, generate_scenario


# ======================================================================
# SETTINGS
# ======================================================================
RUN_ID = "ts3-run_003"
SAVE_ROOT = "checkpoints"
NUM_SCENARIOS = 10
START_SEED = 30000
N_TASKS = None
PREFERENCE = (0.2, 0.8)  # [lambda_E, lambda_T]
GREEDY = True

# Hold evaluation behavior constant across checkpoints. Set these to None to
# use the corresponding values saved in each checkpoint instead.
PREF_SWITCH_MAX_INJECTIONS = 3
PREF_SWITCH_TASK_THRESHOLD = 12

# Balanced test set: category counts differ by at most one when possible.
BALANCE_TEST_CATEGORIES = True

SAVE_FIGURES = True
SHOW_FIGURES = False
OUTPUT_ROOT = "checkpoint_comparisons"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, SAVE_ROOT, RUN_ID)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_ROOT, RUN_ID)


def checkpoint_iteration(path):
    match = re.search(rf"{re.escape(RUN_ID)}_it(\d+)_ep(\d+)\.pt$", os.path.basename(path))
    if match is None:
        raise ValueError(f"Checkpoint filename does not contain its iteration: {path}")
    return int(match.group(1))


def find_checkpoints():
    if not os.path.isdir(CHECKPOINT_DIR):
        raise FileNotFoundError(f"Checkpoint directory not found: {CHECKPOINT_DIR}")

    prefix = re.escape(RUN_ID)
    paths = [
        os.path.join(CHECKPOINT_DIR, filename)
        for filename in os.listdir(CHECKPOINT_DIR)
        if re.fullmatch(rf"{prefix}_it\d+_ep\d+\.pt", filename)
    ]
    paths.sort(key=checkpoint_iteration)
    if not paths:
        raise FileNotFoundError(f"No saved checkpoints found in {CHECKPOINT_DIR}")
    return paths


def make_scenarios():
    if NUM_SCENARIOS <= 0:
        raise ValueError("NUM_SCENARIOS must be greater than zero")
    if N_TASKS is not None and N_TASKS <= 0:
        raise ValueError("N_TASKS must be greater than zero or None")
    if len(PREFERENCE) != 2 or min(PREFERENCE) < 0 or sum(PREFERENCE) <= 0:
        raise ValueError("PREFERENCE must contain two non-negative values, not both zero")

    if BALANCE_TEST_CATEGORIES:
        categories = tp.balanced_category_schedule(NUM_SCENARIOS, START_SEED)
    else:
        categories = [None] * NUM_SCENARIOS

    scenarios = []
    for index, category in enumerate(categories):
        seed = START_SEED + index
        kwargs = {"return_meta": True}
        if category is not None:
            kwargs["robot_category"] = category
        robots, pos, payload, service, _, error = generate_scenario(
            1, N_TASKS, seed=seed, **kwargs)
        if error is not None:
            raise RuntimeError(f"Could not generate test scenario for seed {seed}: {error}")
        scenarios.append({
            "seed": seed,
            "category": int(robots["category"][0]),
            "robots": robots,
            "pos": pos,
            "payload": payload,
            "service": service,
        })
    return scenarios


def evaluate_checkpoint(path, scenarios, preference):
    evaluator.CKPT_PATH = path
    evaluator.GREEDY = GREEDY
    policy = evaluator.load_model()

    if PREF_SWITCH_MAX_INJECTIONS is not None:
        tp.hp.pref_switch_max_injections = PREF_SWITCH_MAX_INJECTIONS
    if PREF_SWITCH_TASK_THRESHOLD is not None:
        tp.hp.pref_switch_task_threshold = PREF_SWITCH_TASK_THRESHOLD

    device = tp.hp.device
    lam = torch.tensor([preference], dtype=torch.float32, device=device)
    iteration = checkpoint_iteration(path)
    records = []

    for scenario in scenarios:
        env = tp.SingleDroneEnv(
            scenario["robots"], scenario["pos"], scenario["payload"],
            scenario["service"], B=1)
        env.reset()
        if not env.ok.any():
            print(f"  seed {scenario['seed']}: no task fits the battery; skipped")
            continue

        task_features = tp.task_features(
            scenario["pos"], scenario["payload"], scenario["service"], env.cap)
        robot_features = tp.robot_features(scenario["robots"])
        tasks, _, valid, _, _ = evaluator.rollout(
            policy, env, task_features, robot_features, lam)
        served_steps = int(valid[:, 0].sum())
        records.append({
            "run_id": RUN_ID,
            "checkpoint": os.path.basename(path),
            "checkpoint_path": path,
            "iteration": iteration,
            "seed": scenario["seed"],
            "category": scenario["category"],
            "n_tasks": env.N,
            "lambda_E": preference[0],
            "lambda_T": preference[1],
            "energy_wh": float(env.E_used[0].item()),
            "time_s": float(env.T_used[0].item()),
            "unserved": int((~env.visited[0]).sum().item()),
            "served_steps": served_steps,
            "task_order": " ".join(map(str, tasks[:served_steps, 0].tolist())),
        })

    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def checkpoint_labels(records):
    by_iteration = sorted({record["iteration"] for record in records})
    return {iteration: f"it{iteration:05d}" for iteration in by_iteration}


def plot_scenario(records, seed, labels):
    ordered = sorted(records, key=lambda row: row["iteration"])
    x = np.arange(len(ordered))
    tick_labels = [labels[row["iteration"]] for row in ordered]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    metrics = [
        ("energy_wh", "Energy used [Wh]", "tab:blue"),
        ("time_s", "Mission time [s]", "tab:orange"),
        ("unserved", "Unserved tasks", "tab:red"),
    ]
    for ax, (key, title, color) in zip((axes[0, 0], axes[0, 1], axes[1, 0]), metrics):
        values = [row[key] for row in ordered]
        ax.bar(x, values, color=color, alpha=0.82)
        ax.set_title(title)
        ax.set_xticks(x, tick_labels, rotation=60, ha="right")
        ax.grid(axis="y", alpha=0.25)

    scatter = axes[1, 1].scatter(
        [row["time_s"] for row in ordered],
        [row["energy_wh"] for row in ordered],
        c=[row["iteration"] for row in ordered],
        cmap="viridis",
        s=80,
        edgecolors="black",
    )
    for row in ordered:
        axes[1, 1].annotate(
            labels[row["iteration"]],
            (row["time_s"], row["energy_wh"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1, 1].set_xlabel("Mission time [s] (lower is better)")
    axes[1, 1].set_ylabel("Energy used [Wh] (lower is better)")
    axes[1, 1].set_title("Energy / time trade-off")
    axes[1, 1].grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes[1, 1], label="Training iteration")

    first = ordered[0]
    fig.suptitle(
        f"{RUN_ID} checkpoint comparison — seed {seed}, category {first['category']}, "
        f"{first['n_tasks']} tasks, λ=({first['lambda_E']:.2f}, {first['lambda_T']:.2f})"
    )
    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(os.path.join(OUTPUT_DIR, f"scenario_seed{seed}.png"),
                    dpi=160, bbox_inches="tight")
    return fig


def plot_summary(records, labels):
    iterations = sorted({row["iteration"] for row in records})
    grouped = [
        [row for row in records if row["iteration"] == iteration]
        for iteration in iterations
    ]
    names = [labels[iteration] for iteration in iterations]
    means = {
        key: [float(np.mean([row[key] for row in group])) for group in grouped]
        for key in ("energy_wh", "time_s", "unserved")
    }
    stds = {
        key: [float(np.std([row[key] for row in group])) for group in grouped]
        for key in ("energy_wh", "time_s", "unserved")
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, key, title, color in (
        (axes[0, 0], "energy_wh", "Mean energy [Wh]", "tab:blue"),
        (axes[0, 1], "time_s", "Mean mission time [s]", "tab:orange"),
        (axes[1, 0], "unserved", "Mean unserved tasks", "tab:red"),
    ):
        ax.errorbar(
            iterations, means[key], yerr=stds[key], fmt="o-", capsize=3,
            color=color, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("Training iteration")
        ax.set_xticks(iterations, names, rotation=60, ha="right")
        ax.grid(alpha=0.25)

    scatter = axes[1, 1].scatter(
        means["time_s"], means["energy_wh"], c=iterations, cmap="viridis",
        s=100, edgecolors="black")
    for iteration, time_s, energy_wh in zip(iterations, means["time_s"], means["energy_wh"]):
        axes[1, 1].annotate(
            labels[iteration], (time_s, energy_wh),
            xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1, 1].set_xlabel("Mean mission time [s] (lower is better)")
    axes[1, 1].set_ylabel("Mean energy [Wh] (lower is better)")
    axes[1, 1].set_title("Mean energy / time trade-off")
    axes[1, 1].grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes[1, 1], label="Training iteration")
    fig.suptitle(
        f"{RUN_ID}: aggregate over {len(grouped[0])} common scenarios "
        f"at λ=({PREFERENCE[0]:.2f}, {PREFERENCE[1]:.2f})"
    )
    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(os.path.join(OUTPUT_DIR, "aggregate_summary.png"),
                    dpi=160, bbox_inches="tight")
    return fig


def save_results(records):
    path = os.path.join(OUTPUT_DIR, "checkpoint_comparison.csv")
    columns = [
        "run_id", "checkpoint", "checkpoint_path", "iteration", "seed", "category",
        "n_tasks", "lambda_E", "lambda_T", "energy_wh", "time_s", "unserved",
        "served_steps", "task_order",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved results: {path}")


def main():
    checkpoints = find_checkpoints()
    scenarios = make_scenarios()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preference_sum = sum(PREFERENCE)
    preference = (PREFERENCE[0] / preference_sum, PREFERENCE[1] / preference_sum)

    print(f"Run: {RUN_ID}")
    print(f"Checkpoints: {len(checkpoints)}")
    print(f"Scenarios: {len(scenarios)} (same seeds for every checkpoint)")
    print(f"Preference: {preference}; greedy={GREEDY}")
    if BALANCE_TEST_CATEGORIES:
        counts = np.bincount(
            [scenario["category"] for scenario in scenarios],
            minlength=len(CATEGORIES["payload_kg"]))
        print(f"Test scenarios by category: {counts.tolist()}")

    results = []
    for checkpoint_index, path in enumerate(checkpoints, start=1):
        print(f"\n[{checkpoint_index}/{len(checkpoints)}] {os.path.basename(path)}")
        records = evaluate_checkpoint(path, scenarios, preference)
        results.extend(records)
        means = {
            metric: float(np.mean([row[metric] for row in records]))
            for metric in ("energy_wh", "time_s", "unserved")
        } if records else {}
        print(f"  evaluated {len(records)}/{len(scenarios)} scenarios; mean metrics: {means}")

    if not results:
        raise RuntimeError("No checkpoint/scenario combinations produced results")

    labels = checkpoint_labels(results)
    records_by_seed = {}
    for row in results:
        records_by_seed.setdefault(row["seed"], []).append(row)
    for seed, rows in records_by_seed.items():
        plot_scenario(rows, seed, labels)
    plot_summary(results, labels)
    save_results(results)

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
