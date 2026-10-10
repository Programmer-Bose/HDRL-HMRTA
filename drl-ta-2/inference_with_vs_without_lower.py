"""
inference_with_vs_without_lower.py
------------------------------------
Loads the UPPER layer (assignment) and LOWER layer (sequencer), produces
ONE assignment (upper layer, at your actual preference lam3), and then
sequences that SAME assignment two different ways so you can see exactly
what the lower-layer sequencer buys you:

  WITHOUT lower layer : each robot flies its owned tasks in ascending
                         task-index order at a constant speed (its own
                         category speed_ms) -- the cheap proxy used to
                         train the assignment network fast (see
                         task_assignment_fast.proxy_evaluate_assignments).
                         No neural sequencing at all.

  WITH lower layer     : each robot's SAME task set is re-ordered and
                         re-paced by the trained FiLMPointerPolicy
                         (lower layer), at lam2 derived from your lam3
                         via map_lam3_to_lam2 (same top->bottom preference
                         flow as inference_compare.py).

Everything is a hyperparameter in CFG below -- no terminal args needed.
"""

import sys
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, "/mnt/project")
from scenario_gen import generate_scenario, DEPOTS, CATEGORIES
from drone_energy import DroneEnergyModel
import train_ppo as lower
import train_assignment_ppo as upper

from inference_compare import (
    assign_deterministic, sequence_one_robot, map_lam3_to_lam2, print_result, print_comparison,
)


# ======================================================================
# CONFIG (edit here only)
# ======================================================================
@dataclass
class CFG:
    lower_ckpt: str = "ts3-run_004/ts3-run_004_it05000_ep10000.pt"
    upper_ckpt: str = r"checkpoints_assign_fast\assign-fast-run_002\assign-fast-run_002_it00200.pt"

    n_robots: int = 6
    n_tasks: int = None
    seed: int = 123

    # preference vector for the upper layer (normalised internally, need not sum to 1)
    w_makespan: float = 1.0
    w_soc: float = 0.0
    w_capvar: float = 0.0


cfg = CFG()
device = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================
# WITHOUT lower layer: ascending task-index order, constant category speed
# ======================================================================
def sequence_without_lower(robots, task_pos, task_payload, task_service, assign, label, lam3_display):
    """
    Same shape of result as inference_compare.sequence_assignment, but routes
    each robot's owned tasks (ascending index order) with ONE closed-form
    DroneEnergyModel.tour_energy call at constant speed -- no NN sequencer.
    """
    M = len(robots["category"])
    model = DroneEnergyModel(robots, device=device)
    depot_id = np.asarray(robots["depot_id"])
    depot_all = torch.as_tensor(DEPOTS, dtype=torch.float32, device=device)

    per_robot = []
    for i in range(M):
        idx = np.where(assign == i)[0]                               # ascending by construction
        n = len(idx)
        payload_assigned = float(task_payload[idx].sum()) if n > 0 else 0.0
        payload_frac = payload_assigned / float(robots["payload_kg"][i])

        if n == 0:
            per_robot.append({"robot_id": i, "category": int(robots["category"][i]), "order": [],
                              "E_used": 0.0, "T_used": 0.0, "soc_drop": 0.0, "n_unserved": 0,
                              "payload_assigned": 0.0, "payload_frac": 0.0})
            continue

        depot_xyz = depot_all[depot_id[i]][None, None, :]
        route = torch.as_tensor(task_pos[idx], dtype=torch.float32, device=device)[None, :, :]
        coords = torch.cat([depot_xyz, route, depot_xyz], dim=1)
        payload = torch.as_tensor(task_payload[idx], dtype=torch.float32, device=device)[None, :]
        service = torch.as_tensor(task_service[idx], dtype=torch.float32, device=device)[None, :]
        vmax = float(robots["speed_ms"][i])
        speeds = torch.full((1, n + 1), vmax, device=device)
        ids = torch.full((1,), i, dtype=torch.long, device=device)

        out = model.tour_energy(ids, coords, payload, service, speeds=speeds)
        feasible = bool(out["battery_ok"][0]) and bool(out["payload_ok"][0])
        E_used = float(out["energy_wh"][0]) if feasible else 0.0
        T_used = float(out["time_s"][0]) if feasible else 0.0
        n_unserved = 0 if feasible else n                             # whole route rejected if infeasible
        soc_drop = E_used / float(robots["battery_wh"][i]) * 100.0

        per_robot.append({"robot_id": i, "category": int(robots["category"][i]),
                          "order": list(idx), "E_used": E_used, "T_used": T_used,
                          "soc_drop": soc_drop, "n_unserved": n_unserved,
                          "payload_assigned": payload_assigned, "payload_frac": payload_frac})

    unserved_total = sum(r["n_unserved"] for r in per_robot) + int((assign < 0).sum())
    makespan = max((r["T_used"] for r in per_robot), default=0.0)
    max_soc_drop = max((r["soc_drop"] for r in per_robot), default=0.0)
    cap_var = float(np.var([r["payload_frac"] for r in per_robot])) if M > 0 else 0.0

    return {
        # lam2 is not meaningful here (no energy/time trade-off is made -- constant speed,
        # fixed order); use NaN placeholders so print_result's formatting doesn't choke on None.
        "label": label, "lam3": lam3_display, "lam2": [float("nan"), float("nan")], "assign": assign,
        "per_robot": per_robot, "makespan_s": makespan, "max_soc_drop_pct": max_soc_drop,
        "cap_variance": cap_var, "total_unserved": unserved_total,
    }


# ======================================================================
# WITH lower layer: NN sequencer at lam2 derived from lam3 (reuses inference_compare helpers)
# ======================================================================
def sequence_with_lower(lower_policy, robots, task_pos, task_payload, task_service, assign,
                        lam2, label, lam3_display):
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
        "label": label, "lam3": lam3_display, "lam2": lam2, "assign": assign,
        "per_robot": per_robot, "makespan_s": makespan, "max_soc_drop_pct": max_soc_drop,
        "cap_variance": cap_var, "total_unserved": unserved_total,
    }


# ======================================================================
# Main
# ======================================================================
def main():
    # ---- load lower layer (sequencer) -------------------------------------
    lower_policy = lower.FiLMPointerPolicy().to(device)
    lckpt = torch.load(cfg.lower_ckpt, map_location=device, weights_only=True)
    lower_policy.load_state_dict(lckpt["model"])
    lower_policy.eval()
    for p in lower_policy.parameters():
        p.requires_grad_(False)

    # ---- load upper layer (assignment) -------------------------------------
    upper_policy = upper.AssignmentPolicy().to(device)
    uckpt = torch.load(cfg.upper_ckpt, map_location=device, weights_only=True)
    upper_policy.load_state_dict(uckpt["model"])
    upper_policy.eval()
    for p in upper_policy.parameters():
        p.requires_grad_(False)

    # ---- scenario -------------------------------------------------------------
    robots, task_pos, task_payload, task_service, _, err = generate_scenario(
        cfg.n_robots, cfg.n_tasks, seed=cfg.seed, return_meta=True)
    if err is not None:
        print(f"Scenario error: {err}")
        return
    print(f"Scenario: {len(robots['category'])} robots, {len(task_payload)} tasks, seed={cfg.seed}")
    task_order = np.random.default_rng(cfg.seed).permutation(len(task_payload))

    # ---- ONE assignment, from the upper layer at your actual preference ------
    lam3 = [cfg.w_makespan, cfg.w_soc, cfg.w_capvar]
    s = sum(lam3)
    lam3 = [w / s for w in lam3] if s > 0 else [1 / 3, 1 / 3, 1 / 3]
    assign = assign_deterministic(upper_policy, robots, task_pos, task_payload, task_service,
                                   lam3, task_order, device)
    print(f"Preference vector lam3 = {np.round(lam3, 3).tolist()} "
          f"(single assignment, both runs below sequence it)")

    # ---- WITHOUT lower layer: ascending order + constant speed --------------
    without = sequence_without_lower(robots, task_pos, task_payload, task_service, assign,
                                     "WITHOUT LOWER LAYER (ascending order, constant speed)", lam3)

    # ---- WITH lower layer: NN sequencer at lam2 derived from lam3 -----------
    lam2 = map_lam3_to_lam2(lam3)
    with_lower = sequence_with_lower(lower_policy, robots, task_pos, task_payload, task_service,
                                     assign, lam2, "WITH LOWER LAYER (NN sequencer)", lam3)

    # ---- report ------------------------------------------------------------------
    print_result(without)
    print_result(with_lower)
    print_comparison(without, with_lower)


if __name__ == "__main__":
    main()
