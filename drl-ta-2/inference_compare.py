"""
inference_compare.py
---------------------
Side-by-side inference test for the two-stage (assignment -> sequencing)
pipeline:

  BALANCED RUN   : upper-layer assignment at lam3 = [1/3, 1/3, 1/3] (neutral),
                    then each robot's subset sequenced by the frozen lower
                    layer at lam2 = [0.5, 0.5] (exactly how the assignment
                    network was trained to be evaluated).

  PREFERENCE RUN : upper-layer assignment at the user's actual lam3, then
                    each robot's subset sequenced by the lower layer at a
                    lam2 DERIVED FROM lam3 (see map_lam3_to_lam2 below) --
                    NOT the fixed balanced [0.5, 0.5] used during training.

PREFERENCE FLOW (top -> bottom), the important part:
  lam3 = [w_makespan, w_max_soc_drop, w_cap_variance]   (upper layer, 3-d)
  The lower layer only understands a 2-d energy/time trade-off:
      lam2 = [lambda_E, lambda_T],  lambda_E + lambda_T = 1
  w_cap_variance has no lower-layer analogue (it only shapes WHICH tasks
  go to WHICH robot, not how a single robot paces its own route), so it is
  dropped when deriving lam2; the remaining two weights are renormalised:
      lambda_T = w_makespan / (w_makespan + w_max_soc_drop)
      lambda_E = w_max_soc_drop / (w_makespan + w_max_soc_drop)
  This keeps the user's RELATIVE emphasis on time vs. energy intact as it
  flows from the 3-d assignment preference down into the 2-d sequencing
  preference. See map_lam3_to_lam2().

Usage:
    Edit the CFG block below (checkpoint paths, scenario size, preference
    weights) and run:   python inference_compare.py
    No command-line flags needed -- everything is a hyperparameter here.
"""

import sys
from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical

# sys.path.insert(0, "/mnt/project")
from scenario_gen import generate_scenario, DEPOTS, CATEGORIES
import train_ppo as lower
import train_assignment_ppo as upper






# ======================================================================
# CONFIG  (edit here only -- no terminal args needed)
# ======================================================================
@dataclass
class CFG:
    # ---- checkpoints -------------------------------------------------
    lower_ckpt: str = "ts3-run_004/ts3-run_004_it05000_ep10000.pt"   # frozen sequencer
    upper_ckpt: str = "assign-run_001_it00100.pt"  # trained assignment net

    # ---- scenario ------------------------------------------------------
    n_robots: int = 6
    n_tasks: int = None            # None = scenario_gen default per-category range
    seed: int = 1

    # ---- preference vector (upper layer, 3-d; need not pre-sum to 1 -- normalised internally)
    w_makespan: float = 0
    w_soc: float = 1
    w_capvar: float = 0



cfg = CFG()


# ======================================================================
# Preference flow: upper (3-d) -> lower (2-d)
# ======================================================================
def map_lam3_to_lam2(lam3, eps=1e-8):
    """
    lam3 : [w_makespan, w_max_soc_drop, w_cap_variance] (numpy / list, sums to 1 ideally)
    Returns lam2 : [lambda_E, lambda_T] for the lower-layer sequencer.

    w_cap_variance is dropped (it has no sequencing-level meaning); the
    remaining two weights are renormalised so their RATIO -- the user's
    actual emphasis on time vs. energy -- is preserved exactly.
    """
    w_makespan, w_soc, _w_capvar = lam3
    denom = max(w_makespan + w_soc, eps)
    lambda_T = w_makespan / denom
    lambda_E = w_soc / denom
    return [lambda_E, lambda_T]


# ======================================================================
# Deterministic (argmax) assignment pass
# ======================================================================
@torch.no_grad()
def assign_deterministic(policy, robots, task_pos, task_payload, task_service, lam3, task_order, device):
    """One deterministic (argmax) assignment episode. Returns assign [N] long numpy (-1 = unassigned)."""
    M = len(robots["category"])
    N = len(task_payload)
    task_f = upper.assignment_features(task_pos, task_payload, task_service)
    robot_f_static = upper.robot_static_features(robots)
    cap = torch.as_tensor(robots["payload_kg"], dtype=torch.float32, device=device)
    pay = torch.as_tensor(task_payload, dtype=torch.float32, device=device)
    remaining = cap.clone().unsqueeze(0)                              # [1,M]
    lam3_t = torch.as_tensor([lam3], dtype=torch.float32, device=device)

    assign = torch.full((1, N), -1, dtype=torch.long, device=device)
    for j in task_order:
        cap_frac = (remaining / cap[None, :]).clamp(min=0.0)
        Ht, Hr, e = policy.encode(task_f, robot_f_static, cap_frac, lam3_t)
        logits = policy.step_logits(Ht, Hr, e, j)
        feasible = remaining >= pay[j]
        masked_logits = logits.masked_fill(~feasible, -1e9)
        if not feasible.any():
            continue                                                  # task left unassigned (infeasible everywhere)
        choice = masked_logits.argmax(-1)                              # [1]
        assign[0, j] = choice
        remaining[0, choice] -= pay[j]

    return assign[0].cpu().numpy()


# ======================================================================
# Deterministic per-robot sequencing at an ARBITRARY lam2 (not fixed balanced)
# ======================================================================
@torch.no_grad()
def sequence_one_robot(lower_policy, robot_row, pos_sub, pay_sub, srv_sub, lam2, device):
    """
    Deterministically sequence one robot's task subset at the given lam2 = [lambda_E, lambda_T].
    Mirrors task_assignment.FrozenSequencer.rollout_one_robot but with a CONFIGURABLE lam2
    (that function is hardcoded to balanced [0.5, 0.5] on purpose, for training-time stability).

    Returns dict: order (task indices in the order served), E_used (Wh), T_used (s),
                  soc_drop (%), n_unserved (int).
    """
    n = len(pay_sub)
    if n == 0:
        return {"order": [], "E_used": 0.0, "T_used": 0.0, "soc_drop": 0.0, "n_unserved": 0}

    env = lower.SingleDroneEnv(robot_row, pos_sub, pay_sub, srv_sub, B=1)
    env.reset()
    if not env.ok.any():
        return {"order": [], "E_used": 0.0, "T_used": 0.0, "soc_drop": 0.0, "n_unserved": n}

    task_f = lower.task_features(pos_sub, pay_sub, srv_sub, env.cap)
    robot_f = lower.robot_features(robot_row)
    lam = torch.tensor([lam2], dtype=torch.float32, device=device)
    order = []

    for _ in range(n):
        if not env.active.any():
            break
        state = env.observe()
        task, x, speed_frac, x_ret, _, _, _ = lower_policy.act(
            task_f, robot_f, state, lam, env.ok, env.speed_feat,
            env.load / env.cap, env.T_used / 3600.0, env.depot, deterministic=True)
        order.append(int(task.item()))
        env.step(task, speed_frac, x_ret)

    soc_drop = float(env.E_used.item()) / float(robot_row["battery_wh"][0]) * 100.0
    n_unserved = int((~env.visited).sum().item())
    return {"order": order, "E_used": float(env.E_used.item()), "T_used": float(env.T_used.item()),
            "soc_drop": soc_drop, "n_unserved": n_unserved}


# ======================================================================
# Sequence a FIXED assignment (from upper layer) at a given lam2 -> metrics
# ======================================================================
def sequence_assignment(lower_policy, robots, task_pos, task_payload, task_service,
                        assign, lam2, device, label, lam3_for_display):
    """assign is NOT recomputed here -- it is the one fixed partition passed in."""
    M = len(robots["category"])

    per_robot = []
    for i in range(M):
        idx = np.where(assign == i)[0]
        robot_row = {k: np.asarray(v)[i:i + 1] for k, v in robots.items()}
        res = sequence_one_robot(lower_policy, robot_row,
                                 task_pos[idx], task_payload[idx], task_service[idx], lam2, device)
        res["robot_id"] = i
        res["category"] = int(robots["category"][i])
        res["payload_assigned"] = float(task_payload[idx].sum())
        res["payload_frac"] = res["payload_assigned"] / float(robots["payload_kg"][i])
        per_robot.append(res)

    unserved_total = sum(r["n_unserved"] for r in per_robot) + int((assign < 0).sum())
    makespan = max((r["T_used"] for r in per_robot), default=0.0)
    max_soc_drop = max((r["soc_drop"] for r in per_robot), default=0.0)
    cap_var = float(np.var([r["payload_frac"] for r in per_robot])) if M > 0 else 0.0

    return {
        "label": label, "lam3": lam3_for_display, "lam2": lam2, "assign": assign,
        "per_robot": per_robot, "makespan_s": makespan, "max_soc_drop_pct": max_soc_drop,
        "cap_variance": cap_var, "total_unserved": unserved_total,
    }


# ======================================================================
# Pretty printing
# ======================================================================
def print_result(res):
    print(f"\n--- {res['label']} "
          f"(lam3={np.round(res['lam3'], 3).tolist()}, lam2[E,T]={np.round(res['lam2'], 3).tolist()}) ---")
    print(f"  makespan        : {res['makespan_s']:8.1f} s")
    print(f"  max SOC drop    : {res['max_soc_drop_pct']:8.2f} %")
    print(f"  cap. variance   : {res['cap_variance']:8.5f}")
    print(f"  unserved tasks  : {res['total_unserved']:8d}")
    print(f"  {'robot':>5} {'cat':>3} {'n_tasks':>7} {'payload%':>9} {'E (Wh)':>8} {'T (s)':>8} {'SOC%':>6}")
    for r in res["per_robot"]:
        print(f"  {r['robot_id']:>5} {r['category']:>3} {len(r['order']):>7} "
              f"{100 * r['payload_frac']:>8.1f}% {r['E_used']:>8.1f} {r['T_used']:>8.0f} {r['soc_drop']:>6.1f}")


def print_comparison(balanced, tuned):
    print("\n==================== SIDE-BY-SIDE COMPARISON ====================")
    rows = [
        ("makespan (s)", balanced["makespan_s"], tuned["makespan_s"]),
        ("max SOC drop (%)", balanced["max_soc_drop_pct"], tuned["max_soc_drop_pct"]),
        ("cap. variance", balanced["cap_variance"], tuned["cap_variance"]),
        ("unserved tasks", balanced["total_unserved"], tuned["total_unserved"]),
    ]
    print(f"  {'metric':<20}{'balanced':>14}{'preference-tuned':>20}{'delta':>12}")
    for name, b, t in rows:
        print(f"  {name:<20}{b:>14.3f}{t:>20.3f}{(t - b):>12.3f}")


# ======================================================================
# Main
# ======================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- load frozen lower-layer sequencer -------------------------------
    lower_policy = lower.FiLMPointerPolicy().to(device)
    ckpt = torch.load(cfg.lower_ckpt, map_location=device, weights_only=True)
    lower_policy.load_state_dict(ckpt["model"])
    lower_policy.eval()
    for p in lower_policy.parameters():
        p.requires_grad_(False)

    # ---- load trained upper-layer assignment policy -----------------------
    upper_policy = upper.AssignmentPolicy().to(device)
    uckpt = torch.load(cfg.upper_ckpt, map_location=device, weights_only=True)
    upper_policy.load_state_dict(uckpt["model"])
    upper_policy.eval()
    for p in upper_policy.parameters():
        p.requires_grad_(False)

    # ---- scenario -----------------------------------------------------------
    robots, task_pos, task_payload, task_service, _, err = generate_scenario(
        cfg.n_robots, cfg.n_tasks, seed=cfg.seed, return_meta=True)
    if err is not None:
        print(f"Scenario error: {err}")
        return
    print(f"Scenario: {len(robots['category'])} robots, {len(task_payload)} tasks, seed={cfg.seed}")
    task_order = np.random.default_rng(cfg.seed).permutation(len(task_payload))

    # ---- STEP 1: upper layer assigns tasks -> robots, ONCE, at the user's actual lam3 ----
    lam3_user = [cfg.w_makespan, cfg.w_soc, cfg.w_capvar]
    s = sum(lam3_user)
    lam3_user = [w / s for w in lam3_user] if s > 0 else [1 / 3, 1 / 3, 1 / 3]  # normalise to the simplex
    assign = assign_deterministic(upper_policy, robots, task_pos, task_payload, task_service,
                                   lam3_user, task_order, device)
    print(f"Preference vector lam3 = {np.round(lam3_user, 3).tolist()}  "
          f"(this is the ONLY assignment computed -- both runs below sequence it)")

    # ---- STEP 2a: that SAME assignment, sequenced BALANCED (lam2 = [0.5, 0.5]) --------
    lam2_balanced = [0.5, 0.5]
    balanced = sequence_assignment(lower_policy, robots, task_pos, task_payload, task_service,
                                   assign, lam2_balanced, device, "BALANCED SEQUENCING", lam3_user)

    # ---- STEP 2b: that SAME assignment, sequenced at lam2 DERIVED from lam3 ------------
    lam2_user = map_lam3_to_lam2(lam3_user)                              # <-- the preference flow
    tuned = sequence_assignment(lower_policy, robots, task_pos, task_payload, task_service,
                                assign, lam2_user, device, "PREFERENCE-TUNED SEQUENCING", lam3_user)

    # ---- report ---------------------------------------------------------------
    print_result(balanced)
    print_result(tuned)
    print_comparison(balanced, tuned)


if __name__ == "__main__":
    main()