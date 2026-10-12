"""
test_inference_flowthrough.py
------------------------------
End-to-end inference test for the FLOW-THROUGH upper-layer assignment policy
(train_upper_assignment_flowthrough.py). For a given seed and robot count, a
scenario (task count included) is generated automatically, then for each
preference vector in TestHP.upper_lambda_set the trained upper-layer policy
produces a task->robot assignment, which is evaluated by running the frozen
lower-layer sequencer per robot -- EACH ROBOT QUERIED WITH THE LOWER-LAYER
LAMBDA DERIVED FROM THAT SAME UPPER PREFERENCE (flow-through, not balanced).

No per-robot run metrics (energy/time/SoC) are printed here (unlike the
earlier comparison script) -- but each CSV row DOES include: the fleet's
robot categories (type per robot index), the full task->robot assignment
array, and an explicit per-robot task list (which task numbers each robot
was given), alongside the fleet-level summary metrics for that preference
vector. No chat output.

Edit the TEST_HP block below and run:
    python test_inference_flowthrough.py

Output CSV name is built from n_robots / n_tasks / seed, e.g.:
    inference_R5_N40_seed123.csv
"""

import os
import csv
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Tuple

from scenario_gen import generate_scenario
from train_upper_assignment_IP import (
    UpperHP, hp, AssignmentPolicy, AssignmentEnv, FrozenSequencer,
    assign_task_features, assign_robot_features, upper_to_lower_lambda,
    compute_upper_reward,
)


# ======================================================================
# TEST HYPERPARAMETERS  (edit here only)
# ======================================================================
@dataclass
class TestHP:
    n_robots: int = 10
    seed: int = 25                      # n_tasks is NOT set here -- it comes from the
                                          # scenario generator automatically, given the seed

    # Any number of (w_makespan, w_minSoC, w_payloadVar) vectors; each is auto-normalised.
    upper_lambda_set: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (1.0, 0.0, 0.0),     # makespan-only
        (0.0, 1.0, 0.0),     # min-SoC-only
        # (0.0, 0.0, 1.0),     # payload-balance-only
        (0.6, 0.3, 0.1),
        (0.34, 0.33, 0.33),  # near-balanced
    ])

    upper_ckpt: str = r"checkpoints\upper-assign-run_004-FTIP\upper-assign-run_004-FTIP_it00400.pt"
    lower_run_dir: str = hp.lower_run_dir
    lower_ckpt_path: str = hp.lower_ckpt_path   # "" = latest checkpoint in lower_run_dir

    device: str = hp.device
    out_dir: str = "output"                   # directory the CSV is written into


thp = TestHP()


# ------------------------------------------------------------------
def load_upper_policy(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    policy = AssignmentPolicy().to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    return policy


def run_assignment(policy, robots, task_pos, task_payload, task_service, device, upper_lambda):
    """Deterministic rollout of the trained upper-layer policy -> assign array [N]."""
    N = len(task_payload)
    fleet_cap = robots["payload_kg"].sum()
    task_f = assign_task_features(task_pos, task_payload, task_service, fleet_cap)
    robot_f = assign_robot_features(robots)
    cap_t = torch.as_tensor(robots["payload_kg"], dtype=torch.float32, device=device)

    env = AssignmentEnv(robots, task_pos, task_payload, task_service, B=1)
    env.reset()
    with torch.no_grad():
        H = policy.encode_tasks(task_f)
    lam = torch.as_tensor([upper_lambda], dtype=torch.float32, device=device)

    for step in range(N):
        robot_mask = env.mask(step)
        assign_state = env.observe()
        with torch.no_grad():
            choice, _, _, _ = policy.act(H, step, robot_f, env.remaining_cap, cap_t,
                                         assign_state, lam, robot_mask, deterministic=True)
        env.step(step, choice)
    return env.assign[0].cpu().numpy(), lam                 # [N] robot id per task, [1,3] lam tensor


def rollout_fleet_flowthrough(sequencer, robots, task_pos, task_payload, task_service, assign,
                              lam_lower, deterministic=True):
    """Run every robot's assigned subset through the sequencer at the FLOW-THROUGH lam_lower.
    Returns the 3 fleet-level objective values (no per-robot breakdown)."""
    n_robots = len(robots["category"])
    E_list, T_list, uns_list, usable_list, batt_list = [], [], [], [], []
    assigned_load = torch.zeros(n_robots)

    for r in range(n_robots):
        idx = np.where(assign == r)[0]
        robot_r = {k: v[r:r + 1] for k, v in robots.items()}
        sub_pos, sub_pay, sub_srv = task_pos[idx], task_payload[idx], task_service[idx]
        E, T, uns, usable, batt = sequencer.rollout_robot(
            robot_r, sub_pos, sub_pay, sub_srv, lam_lower, deterministic=deterministic)
        E_list.append(E); T_list.append(T); uns_list.append(uns)
        usable_list.append(usable); batt_list.append(batt)
        assigned_load[r] = float(sub_pay.sum())

    cap_t = torch.as_tensor(robots["payload_kg"], dtype=torch.float32)
    # lam passed to compute_upper_reward only scales a scalar "reward" we don't need here;
    # pass a neutral [1,1,1] since we only use the returned raw metrics, not the reward itself.
    _, makespan, min_soc, payload_var, total_unserved, idle_frac = compute_upper_reward(
        E_list, T_list, uns_list, usable_list, batt_list,
        assigned_load, cap_t, [1.0, 1.0, 1.0], len(task_payload))

    unassigned = int((assign == -1).sum())
    total_unserved_all = total_unserved + unassigned
    return dict(makespan=makespan, min_soc=min_soc, payload_var=payload_var,
               unserved=total_unserved_all, idle_frac=idle_frac)


# ------------------------------------------------------------------
def main():
    device = thp.device

    robots, task_pos, task_payload, task_service, _, err = generate_scenario(
        thp.n_robots, None, seed=thp.seed, return_meta=True)      # n_tasks auto-generated
    if err is not None:
        raise RuntimeError(f"Scenario error: {err}")

    n_tasks = len(task_payload)
    out_name = f"inference_R{thp.n_robots}_N{n_tasks}_seed{thp.seed}.csv"
    out_path = os.path.join(thp.out_dir, out_name)

    upper_policy = load_upper_policy(thp.upper_ckpt, device)
    sequencer = FrozenSequencer(thp.lower_run_dir, thp.lower_ckpt_path, device)

    robot_categories = robots["category"].tolist()             # type (0-3) per robot index

    rows = []
    for raw_lambda in thp.upper_lambda_set:
        upper_lambda = (np.array(raw_lambda) / sum(raw_lambda)).tolist()

        assign, lam_t = run_assignment(upper_policy, robots, task_pos, task_payload,
                                       task_service, device, upper_lambda)
        lam_lower = upper_to_lower_lambda(lam_t)[0].cpu().tolist()   # [energy, time], flow-through

        metrics = rollout_fleet_flowthrough(sequencer, robots, task_pos, task_payload,
                                            task_service, assign, lam_lower)

        # per-robot task list: {robot_id: [task ids assigned to it]}, in task order
        robot_tasks = {r: [] for r in range(thp.n_robots)}
        for task_id, robot_id in enumerate(assign.tolist()):
            if robot_id >= 0:
                robot_tasks[robot_id].append(task_id)
        unassigned_tasks = [t for t, r in enumerate(assign.tolist()) if r < 0]

        # human-readable "R0(cat2):[1,3,7]; R1(cat0):[]; ..." summary, one entry per robot
        robot_tasks_str = "; ".join(
            f"R{r}(cat{robot_categories[r]}):{robot_tasks[r]}" for r in range(thp.n_robots))

        rows.append(dict(
            n_robots=thp.n_robots, n_tasks=n_tasks, seed=thp.seed,
            robot_categories=str(robot_categories),
            w_makespan=round(upper_lambda[0], 4), w_minsoc=round(upper_lambda[1], 4),
            w_payloadvar=round(upper_lambda[2], 4),
            lam_lower_energy=round(lam_lower[0], 4), lam_lower_time=round(lam_lower[1], 4),
            assignment=str(assign.tolist()),
            robot_task_lists=robot_tasks_str,
            unassigned_tasks=str(unassigned_tasks),
            makespan_s=round(metrics["makespan"], 2), min_soc_pct=round(metrics["min_soc"], 2),
            payload_var=round(metrics["payload_var"], 5), unserved=metrics["unserved"],
            idle_frac=round(metrics["idle_frac"], 4),
        ))

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()