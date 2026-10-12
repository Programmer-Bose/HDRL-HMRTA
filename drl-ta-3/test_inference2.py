"""
test_inference.py
------------------
Compare upper-layer assignment outcomes under two lower-layer regimes, for a
SET of upper-layer preference vectors, in one run:
  (a) BALANCED   lower-layer preference, lambda_lower = [0.5, 0.5]        (how it was trained)
  (b) ACTUAL     lower-layer preference, derived from the upper preference vector
                 lam_upper = [w_makespan, w_minSoC, w_payloadVar]
                 -> lam_lower = normalize([w_minSoC, w_makespan])   # [energy, time]
                 (payload_var has no lower-layer analogue, so it does not flow down)

For every preference vector, the task -> robot assignment is printed as a plain
array where index = task id and value = assigned robot id (-1 = unassigned),
so allocation diversity can be read off directly.

Edit the TEST_HP block below and run:
    python test_inference.py
Output is written to OUT_PATH (see TEST_HP) and NOT printed to chat.
"""

import sys
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Tuple

from scenario_gen import generate_scenario
from drone_energy import RESERVE_SOC
from train_ppo import FiLMPointerPolicy, SingleDroneEnv, \
    task_features as lower_task_features, robot_features as lower_robot_features
from train_upper_assignment import (
    UpperHP, hp, AssignmentPolicy, AssignmentEnv,
    assign_task_features, assign_robot_features, FrozenSequencer,
)


# ======================================================================
# TEST HYPERPARAMETERS  (edit here only)
# ======================================================================
@dataclass
class TestHP:
    n_robots: int = 5
    n_tasks: int = None                    # None = category-driven random count
    seed: int = 55

    # Any number of (w_makespan, w_minSoC, w_payloadVar) vectors; each is auto-normalised.
    upper_lambda_set: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (1.0, 0.0, 0.0),     # makespan-only
        (0.0, 1.0, 0.0),     # min-SoC-only
        (0.0, 0.0, 1.0),     # payload-balance-only
        (0.2, 0.6, 0.2),
        (0.34, 0.33, 0.33),  # near-balanced
    ])

    upper_ckpt: str = "checkpoints/upper-assign-run_002/upper-assign-run_002_it01300.pt"
    lower_run_dir: str = hp.lower_run_dir
    lower_ckpt_path: str = "ts3-run_004/ts3-run_004_it05000_ep10000.pt" 

    device: str = hp.device
    out_path: str = f"output/test_inference_{n_robots}_{n_tasks}_{seed}.txt"



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
    H = policy.encode_tasks(task_f)
    lam = torch.as_tensor([upper_lambda], dtype=torch.float32, device=device)

    for step in range(N):
        robot_mask = env.mask(step)
        assign_state = env.observe()
        with torch.no_grad():
            choice, _, _, _ = policy.act(H, step, robot_f, env.remaining_cap, cap_t,
                                         assign_state, lam, robot_mask, deterministic=True)
        env.step(step, choice)
    return env.assign[0].cpu().numpy()                      # [N] robot id per task (-1 = unassigned)


def rollout_fleet(sequencer, robots, task_pos, task_payload, task_service, assign,
                   lam_lower, deterministic=True):
    """Run every robot's assigned subset through the sequencer at a FIXED lam_lower.
    Returns per-robot metrics + aggregate objective values."""
    n_robots = len(robots["category"])
    rows = []
    for r in range(n_robots):
        idx = np.where(assign == r)[0]
        robot_r = {k: v[r:r + 1] for k, v in robots.items()}
        sub_pos, sub_pay, sub_srv = task_pos[idx], task_payload[idx], task_service[idx]

        if len(idx) == 0:
            battery = float(robot_r["battery_wh"][0])
            E, T, uns = 0.0, 0.0, 0
            usable = (1.0 - RESERVE_SOC) * battery
        else:
            env = SingleDroneEnv(robot_r, sub_pos, sub_pay, sub_srv, B=1)
            env.reset()
            if not env.ok.any():
                battery = float(robot_r["battery_wh"][0])
                E, T, uns = 0.0, 0.0, len(idx)
                usable = (1.0 - RESERVE_SOC) * battery
            else:
                cap = float(robot_r["payload_kg"][0])
                task_f = lower_task_features(sub_pos, sub_pay, sub_srv, cap)
                robot_f = lower_robot_features(robot_r)
                lam_t = torch.tensor([lam_lower], dtype=torch.float32, device=sequencer.device)
                for _ in range(len(idx)):
                    if not env.active.any():
                        break
                    state = env.observe()
                    task, x, speed_frac, x_ret, *_ = sequencer.policy.act(
                        task_f, robot_f, state, lam_t, env.ok, env.speed_feat,
                        env.load / env.cap, env.T_used / 3600.0, env.depot,
                        deterministic=deterministic)
                    env.step(task, speed_frac, x_ret)
                E, T = env.E_used.item(), env.T_used.item()
                uns = int((~env.visited).sum().item())
                usable = env.usable
                battery = float(robot_r["battery_wh"][0])

        soc_left = (usable - E) / battery * 100.0
        load = float(task_payload[np.where(assign == r)[0]].sum())
        rows.append(dict(robot=r, n_tasks=int((assign == r).sum()), payload=load,
                         cap=float(robot_r["payload_kg"][0]), E_wh=E, T_s=T,
                         unserved=uns, soc_left=soc_left))

    makespan = max(r["T_s"] for r in rows)
    min_soc = min(r["soc_left"] for r in rows)
    load_frac = np.array([r["payload"] / r["cap"] for r in rows])
    payload_var = float(np.var(load_frac))
    unassigned = int((assign == -1).sum())
    total_unserved = sum(r["unserved"] for r in rows) + unassigned

    return rows, dict(makespan=makespan, min_soc=min_soc, payload_var=payload_var,
                      unserved=total_unserved)


def fmt_report(label, rows, agg):
    lines = [f"\n--- {label} ---"]
    for r in rows:
        lines.append(f"  R{r['robot']}: {r['n_tasks']} tasks, load {r['payload']:.2f}/{r['cap']:.1f} kg, "
                     f"E {r['E_wh']:.1f} Wh, T {r['T_s']:.0f} s, SoC left {r['soc_left']:.1f}%, "
                     f"unserved {r['unserved']}")
    lines.append(f"  => makespan {agg['makespan']:.0f} s | min SoC {agg['min_soc']:.1f}% | "
                f"payload var {agg['payload_var']:.4f} | total unserved {agg['unserved']}")
    return "\n".join(lines)


def assignment_diversity(assign, n_robots):
    """Count of tasks per robot, for a quick diversity read alongside the raw array."""
    counts = [int((assign == r).sum()) for r in range(n_robots)]
    unassigned = int((assign == -1).sum())
    return counts, unassigned


# ------------------------------------------------------------------
def main():
    out = open(thp.out_path, "w")

    def log(msg=""):
        out.write(str(msg) + "\n")

    device = thp.device

    robots, task_pos, task_payload, task_service, _, err = generate_scenario(
        thp.n_robots, thp.n_tasks, seed=thp.seed, return_meta=True)
    if err is not None:
        log(f"Scenario error: {err}")
        out.close()
        return

    log(f"Scenario: {thp.n_robots} robots, {len(task_payload)} tasks, seed {thp.seed}")

    upper_policy = load_upper_policy(thp.upper_ckpt, device)
    sequencer = FrozenSequencer(thp.lower_run_dir, thp.lower_ckpt_path, device)

    for raw_lambda in thp.upper_lambda_set:
        upper_lambda = (np.array(raw_lambda) / sum(raw_lambda)).tolist()
        w_makespan, w_minsoc, w_payvar = upper_lambda

        lam_balanced = [0.5, 0.5]
        raw_lower = np.array([w_minsoc, w_makespan])
        s = raw_lower.sum()
        if s <= 1e-9:
            # neither energy- nor time-relevant weight is present (e.g. payload-var-only
            # preference): no directional signal to send down, fall back to balanced instead
            # of dividing by zero.
            lam_actual = list(lam_balanced)
            lam_actual_note = " (fallback: w_minsoc=w_makespan=0, no lower-layer signal)"
        else:
            lam_actual = (raw_lower / s).tolist()
            lam_actual_note = ""

        log("\n" + "=" * 80)
        log(f"Upper lambda (makespan, minSoC, payloadVar): {['%.3f' % v for v in upper_lambda]}")
        log(f"-> lower lambda ACTUAL   (energy, time): {['%.3f' % v for v in lam_actual]}")
        log(f"-> lower lambda BALANCED (energy, time): {lam_balanced}")

        assign = run_assignment(upper_policy, robots, task_pos, task_payload, task_service,
                                device, upper_lambda)

        counts, unassigned = assignment_diversity(assign, thp.n_robots)
        log(f"\nTask -> robot assignment array (index = task id, value = robot id, -1 = unassigned):")
        log(assign.tolist())
        log(f"Tasks per robot: {counts}  | unassigned: {unassigned}")

        rows_bal, agg_bal = rollout_fleet(sequencer, robots, task_pos, task_payload, task_service,
                                          assign, lam_balanced)
        rows_act, agg_act = rollout_fleet(sequencer, robots, task_pos, task_payload, task_service,
                                          assign, lam_actual)

        log(fmt_report("BALANCED lower-layer preference (training condition)", rows_bal, agg_bal))
        log(fmt_report("ACTUAL lower-layer preference (upper -> lower flow-through)", rows_act, agg_act))

        log("\n--- Delta (actual - balanced) ---")
        log(f"  makespan: {agg_act['makespan'] - agg_bal['makespan']:+.0f} s")
        log(f"  min SoC : {agg_act['min_soc'] - agg_bal['min_soc']:+.1f} %")
        log(f"  pay var : {agg_act['payload_var'] - agg_bal['payload_var']:+.4f}")
        log(f"  unserved: {agg_act['unserved'] - agg_bal['unserved']:+d}")

    out.close()


if __name__ == "__main__":
    main()