"""
test_inference.py
------------------
Compare upper-layer assignment outcomes under two lower-layer regimes:
  (a) BALANCED   lower-layer preference, lambda_lower = [0.5, 0.5]        (how it was trained)
  (b) ACTUAL     lower-layer preference, derived from the upper preference vector
                 lam_upper = [w_makespan, w_minSoC, w_payloadVar]
                 -> lam_lower = normalize([w_minSoC, w_makespan])   # [energy, time]
                 (payload_var has no lower-layer analogue, so it does not flow down)

Edit the TEST_HP block below and run:
    python test_inference.py
"""

import numpy as np
import torch
from dataclasses import dataclass

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
    n_tasks: int = 20                    # None = category-driven random count
    seed: int = 1730

    upper_lambda: tuple = (0, 1, 0)   # (w_makespan, w_minSoC, w_payloadVar), auto-normalised

    upper_ckpt: str = "checkpoints/upper-assign-run_002/upper-assign-run_002_it02000.pt"
    lower_run_dir: str = hp.lower_run_dir
    lower_ckpt_path: str = "ts3-run_004/ts3-run_004_it05000_ep10000.pt"            # "" = latest checkpoint in lower_run_dir

    device: str = hp.device


thp = TestHP()


# ------------------------------------------------------------------
def derive_lower_lambda(upper_lambda):
    """Map upper-layer SoC/makespan weights to lower-layer energy/time weights."""
    lower_weights = np.array([upper_lambda[1], upper_lambda[0]], dtype=float)
    total = lower_weights.sum()
    if total == 0.0:
        return [0.5, 0.5]
    return (lower_weights / total).tolist()


def load_upper_policy(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    policy = AssignmentPolicy().to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    return policy


def run_assignment(policy, robots, task_pos, task_payload, task_service, device, upper_lambda):
    """Deterministic rollout of the trained upper-layer policy -> assign tensor [N]."""
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


def print_report(label, rows, agg):
    print(f"\n--- {label} ---")
    for r in rows:
        print(f"  R{r['robot']}: {r['n_tasks']} tasks, load {r['payload']:.2f}/{r['cap']:.1f} kg, "
              f"E {r['E_wh']:.1f} Wh, T {r['T_s']:.0f} s, SoC left {r['soc_left']:.1f}%, "
              f"unserved {r['unserved']}")
    print(f"  => makespan {agg['makespan']:.0f} s | min SoC {agg['min_soc']:.1f}% | "
          f"payload var {agg['payload_var']:.4f} | total unserved {agg['unserved']}")


# ------------------------------------------------------------------
def main():
    device = thp.device
    upper_lambda = (np.array(thp.upper_lambda) / sum(thp.upper_lambda)).tolist()
    w_makespan, w_minsoc, w_payvar = upper_lambda

    # two of the three upper weights flow into the lower layer:
    # minSoC -> energy weight, makespan -> time weight (payload_var has no lower analogue)
    lam_actual = derive_lower_lambda(upper_lambda)
    lam_balanced = [0.5, 0.5]

    robots, task_pos, task_payload, task_service, _, err = generate_scenario(
        thp.n_robots, thp.n_tasks, seed=thp.seed, return_meta=True)
    if err is not None:
        print(f"Scenario error: {err}")
        return

    print(f"Scenario: {thp.n_robots} robots, {len(task_payload)} tasks, seed {thp.seed}")
    print(f"Upper lambda (makespan, minSoC, payloadVar): {upper_lambda}")
    print(f"-> lower lambda ACTUAL (energy, time): {lam_actual}")
    print(f"-> lower lambda BALANCED (energy, time): {lam_balanced}")

    upper_policy = load_upper_policy(thp.upper_ckpt, device)
    sequencer = FrozenSequencer(thp.lower_run_dir, thp.lower_ckpt_path, device)

    assign = run_assignment(upper_policy, robots, task_pos, task_payload, task_service,
                            device, upper_lambda)

    rows_bal, agg_bal = rollout_fleet(sequencer, robots, task_pos, task_payload, task_service,
                                      assign, lam_balanced)
    rows_act, agg_act = rollout_fleet(sequencer, robots, task_pos, task_payload, task_service,
                                      assign, lam_actual)

    print_report("BALANCED lower-layer preference (training condition)", rows_bal, agg_bal)
    print_report("ACTUAL lower-layer preference (upper -> lower flow-through)", rows_act, agg_act)

    print("\n--- Delta (actual - balanced) ---")
    print(f"  makespan: {agg_act['makespan'] - agg_bal['makespan']:+.0f} s")
    print(f"  min SoC : {agg_act['min_soc'] - agg_bal['min_soc']:+.1f} %")
    print(f"  pay var : {agg_act['payload_var'] - agg_bal['payload_var']:+.4f}")
    print(f"  unserved: {agg_act['unserved'] - agg_bal['unserved']:+d}")


if __name__ == "__main__":
    main()