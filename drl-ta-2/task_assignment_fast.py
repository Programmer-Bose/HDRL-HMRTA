"""
task_assignment_fast.py
------------------------
Faster upper-layer (assignment) training that does NOT call the lower-layer
neural sequencer at all during training. Instead, each robot's assigned
tasks are evaluated with a cheap, closed-form PROXY:

    - order        : tasks are visited in ASCENDING task-index order
                      (e.g. assigned {4, 2, 5} -> flown as 2, 4, 5)
    - speed         : constant = that robot's category speed_ms (its own
                      top speed -- "category average" since every robot of
                      one category shares the same speed_ms in scenario_gen)
    - energy/time   : one closed-form call to DroneEnergyModel.tour_energy
                      per (episode, robot) -- pure tensor math, no autoregressive
                      network forward passes, no step loop. This is what makes
                      it fast: training no longer pays for N sequential NN
                      calls per robot per episode.

PENALTY: if a robot's proxy tour is battery-infeasible (tour_energy's
`battery_ok` is False -- i.e. this ascending-order / constant-speed route
would drain past the reserve), every task on that route is counted as
"unserved" and charged `ahp.unserved_penalty`, exactly like a genuinely
unassigned task. This discourages the assignment network from overloading
any one robot even though the proxy itself never checks mid-route SOC the
way the real sequencer does.

AFTER training with this fast proxy, use the REAL lower-layer sequencer
(FrozenSequencer / sequence_assignment in task_assignment.py / inference_compare.py)
at INFERENCE time to fine-tune / re-order each robot's tasks for real
energy-aware pacing -- the proxy is only a cheap training-time stand-in.

OUTPUT: get_task_allocation_vector() returns the thing you actually want --
a 1-D array of length N where position = task id and value = assigned
robot id (or -1 if left unserved), straight out of the existing
autoregressive assignment decoder.

Files needed in the same folder (project files, already present):
    scenario_gen.py, drone_energy.py, train_ppo.py, task_assignment.py
"""

import os
import sys
import glob
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# sys.path.insert(0, "/mnt/project")
from scenario_gen import generate_scenario, DEPOTS, CATEGORIES
from drone_energy import DroneEnergyModel, RESERVE_SOC

# reuse the assignment network + rollout machinery already built
from train_assignment_ppo import (
    AssignmentPolicy, assignment_features, robot_static_features,
    rollout_assignment, ahp,
)


# ======================================================================
# Extra hyperparameters specific to fast (proxy-trained) assignment
# ======================================================================
@dataclass
class FASTHP:
    run_id: str = "assign-fast-run_002"
    save_root: str = "checkpoints_assign_fast"
    save_every: int = 50
    resume: bool = True
    resume_path: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_seed: int = 11

    start_seed: int = 1
    num_scenarios: int = 500          # cheap proxy -> can afford more scenarios than the NN-evaluator version
    n_robots_min: int = 4
    n_robots_max: int = 10
    n_tasks: int = None

    rollouts_per_scenario: int = 8    # can go higher than the NN-sequencer version since eval is cheap now


fhp = FASTHP()
dev = fhp.device


# ======================================================================
# Fast proxy evaluator
# ======================================================================
def proxy_evaluate_assignments(robots, task_pos, task_payload, task_service, assign):
    """
    assign : torch.long [B,N]  (robot id per task, -1 = unassigned)

    For every (episode b, robot i), build that robot's route as its owned
    tasks in ASCENDING task-index order, flown at its own constant speed_ms,
    and score it with ONE closed-form DroneEnergyModel.tour_energy call
    (no NN, no step loop). Infeasible routes (battery_ok == False) count
    every task on them as unserved.

    Returns dict of torch tensors, each [B]:
        makespan_s, max_soc_drop_pct, cap_variance, total_unserved
    """
    B, N = assign.shape
    M = len(robots["category"])
    model = DroneEnergyModel(robots, device=dev)

    pos_t = torch.as_tensor(task_pos, dtype=torch.float32, device=dev)
    pay_t = torch.as_tensor(task_payload, dtype=torch.float32, device=dev)
    srv_t = torch.as_tensor(task_service, dtype=torch.float32, device=dev)
    cap_t = torch.as_tensor(robots["payload_kg"], dtype=torch.float32, device=dev)
    battery_t = torch.as_tensor(robots["battery_wh"], dtype=torch.float32, device=dev)
    vmax_t = torch.as_tensor(robots["speed_ms"], dtype=torch.float32, device=dev)
    depot_id = np.asarray(robots["depot_id"])
    depot_all = torch.as_tensor(DEPOTS, dtype=torch.float32, device=dev)

    E_used = torch.zeros(B, M, device=dev)
    T_used = torch.zeros(B, M, device=dev)
    unserved = torch.zeros(B, M, device=dev)
    payload_frac = torch.zeros(B, M, device=dev)

    assign_np = assign.cpu().numpy()
    for b in range(B):
        for i in range(M):
            idx = np.where(assign_np[b] == i)[0]                    # already ASCENDING (np.where order)
            n = len(idx)
            payload_frac[b, i] = pay_t[idx].sum() / cap_t[i] if n > 0 else 0.0
            if n == 0:
                continue
            depot_xyz = depot_all[depot_id[i]][None, None, :]       # [1,1,3]
            route = pos_t[idx][None, :, :]                          # [1,n,3]
            coords = torch.cat([depot_xyz, route, depot_xyz], dim=1)  # [1,n+2,3]
            payload = pay_t[idx][None, :]                           # [1,n]
            service = srv_t[idx][None, :]                           # [1,n]
            speeds = vmax_t[i].expand(1, n + 1)                     # constant category speed, every leg

            ids = torch.full((1,), i, dtype=torch.long, device=dev)
            out = model.tour_energy(ids, coords, payload, service, speeds=speeds)

            if bool(out["battery_ok"][0]) and bool(out["payload_ok"][0]):
                E_used[b, i] = out["energy_wh"][0]
                T_used[b, i] = out["time_s"][0]
            else:
                # infeasible route under the proxy -> every task on it is "unserved"
                unserved[b, i] = n
                E_used[b, i] = out["energy_wh"][0].clamp(min=0)      # keep for reference / logging only
                T_used[b, i] = out["time_s"][0].clamp(min=0)

    unserved_total = unserved.sum(1) + (assign < 0).sum(1).float()
    soc_drop = E_used / battery_t[None, :] * 100.0
    makespan = T_used.max(1).values
    max_soc_drop = soc_drop.max(1).values
    cap_variance = payload_frac.var(1, unbiased=False)

    return {
        "makespan_s": makespan, "max_soc_drop_pct": max_soc_drop,
        "cap_variance": cap_variance, "total_unserved": unserved_total,
    }


def proxy_rewards(robots, task_pos, task_payload, task_service, assign, lam3, T_ref_s, soc_ref_pct=100.0):
    """lam3 : [B,3]. Returns reward [B], metrics dict (see proxy_evaluate_assignments)."""
    metrics = proxy_evaluate_assignments(robots, task_pos, task_payload, task_service, assign)
    makespan_n = metrics["makespan_s"] / T_ref_s
    soc_n = metrics["max_soc_drop_pct"] / soc_ref_pct
    cap_var = metrics["cap_variance"]
    reward = -(lam3[:, 0] * makespan_n + lam3[:, 1] * soc_n + lam3[:, 2] * cap_var) \
             - ahp.unserved_penalty * metrics["total_unserved"]
    return reward, metrics


# ======================================================================
# Task allocation vector (the actual deliverable)
# ======================================================================
def get_task_allocation_vector(assign):
    """
    assign : torch.long [N] (one episode) or [B,N] (batch).
    Returns a plain numpy array, index = task id, value = assigned robot id
    (-1 = left unserved / infeasible for every robot).
    """
    return assign.detach().cpu().numpy()


# ======================================================================
# PPO update (identical scheme to task_assignment.ppo_update, imported logic
# inlined here so this file has no dependency on the NN-sequencer reward path)
# ======================================================================
def ppo_update_fast(policy, optimizer, robots, task_pos, task_payload, task_service, task_order,
                    lam3, assign, old_logp, reward):
    adv = reward - reward.mean()
    if reward.shape[0] > 1:
        adv = adv / (adv.std() + 1e-8)

    stats = []
    for _ in range(ahp.epochs_per_scenario):
        B = lam3.shape[0]
        M = len(robots["category"])
        task_f = assignment_features(task_pos, task_payload, task_service)
        robot_f_static = robot_static_features(robots)
        cap = torch.as_tensor(robots["payload_kg"], dtype=torch.float32, device=dev)
        pay = torch.as_tensor(task_payload, dtype=torch.float32, device=dev)
        remaining = cap[None, :].expand(B, M).clone()

        logp_sum = torch.zeros(B, device=dev)
        ent_sum = torch.zeros(B, device=dev)
        value0 = None
        for step, j in enumerate(task_order):
            cap_frac = (remaining / cap[None, :]).clamp(min=0.0)
            Ht, Hr, e = policy.encode(task_f, robot_f_static, cap_frac, lam3)
            if step == 0:
                value0 = policy.value(Ht, Hr, e)
            logits = policy.step_logits(Ht, Hr, e, j)
            feasible = remaining >= pay[j]
            masked_logits = logits.masked_fill(~feasible, -1e9)
            any_feasible = feasible.any(-1)
            dist = Categorical(logits=masked_logits)
            choice = assign[:, j].clamp(min=0)
            logp = dist.log_prob(choice)
            ent = dist.entropy()
            logp_sum = logp_sum + logp * any_feasible
            ent_sum = ent_sum + ent * any_feasible
            rows = torch.arange(B, device=dev)
            chosen_cap = remaining[rows, choice]
            remaining[rows, choice] = torch.where(any_feasible, chosen_cap - pay[j], chosen_cap)

        ratio = torch.exp(logp_sum - old_logp)
        pg_loss = -torch.min(ratio * adv,
                             torch.clamp(ratio, 1 - ahp.clip_eps, 1 + ahp.clip_eps) * adv).mean()
        v_loss = (value0 - reward).pow(2).mean()
        loss = pg_loss + ahp.value_coef * v_loss - ahp.entropy_coef * ent_sum.mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), ahp.max_grad_norm)
        optimizer.step()
        stats.append([pg_loss.item(), v_loss.item(), ent_sum.mean().item()])

    return np.mean(stats, axis=0)


# ======================================================================
# Checkpoints
# ======================================================================
def save_checkpoint(run_dir, it, policy, optimizer):
    name = f"{fhp.run_id}_it{it:05d}.pt"
    torch.save({"iteration": it, "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(), "hparams": asdict(fhp)},
               os.path.join(run_dir, name))
    return name


def load_checkpoint(run_dir, policy, optimizer):
    path = fhp.resume_path or (sorted(glob.glob(os.path.join(run_dir, f"{fhp.run_id}_it*.pt"))) or [None])[-1]
    if not path:
        print("No fast-assignment checkpoint found -> starting from scratch.")
        return 0
    ckpt = torch.load(path, map_location=dev, weights_only=True)
    policy.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Resumed from {os.path.basename(path)} (iter {ckpt['iteration']}).")
    return ckpt["iteration"]


# ======================================================================
# Training (no NN sequencer involved -- proxy reward only)
# ======================================================================
def train():
    torch.manual_seed(fhp.torch_seed)
    run_dir = os.path.join(fhp.save_root, fhp.run_id)
    os.makedirs(run_dir, exist_ok=True)

    policy = AssignmentPolicy().to(dev)
    optimizer = torch.optim.Adam(policy.parameters(), lr=ahp.lr)
    start_it = load_checkpoint(run_dir, policy, optimizer) if fhp.resume else 0

    log_path = os.path.join(run_dir, "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("iter,seed,n_robots,reward_mean,makespan_s_mean,max_soc_mean,cap_var_mean,"
                    "unserved_mean,pg_loss,v_loss,entropy\n")

    for it in range(start_it, fhp.num_scenarios):
        seed = fhp.start_seed + it
        n_robots_it = int(np.random.default_rng(seed).integers(fhp.n_robots_min, fhp.n_robots_max + 1))
        robots, task_pos, task_payload, task_service, _, err = generate_scenario(
            n_robots_it, fhp.n_tasks, seed=seed, return_meta=True)
        if err is not None or len(task_payload) == 0:
            print(f"[it {it}] seed {seed}: scenario error ({err}) -> skipped")
            continue

        B = fhp.rollouts_per_scenario
        lam3 = torch.distributions.Dirichlet(torch.full((3,), ahp.pref_alpha if hasattr(ahp, "pref_alpha") else 1.0,
                                                         device=dev)).sample((B,))
        T_ref_s = float(np.max(robots["flight_min"])) * 60.0
        task_order = np.random.permutation(len(task_payload))

        with torch.no_grad():
            assign, logp0, _, _ = rollout_assignment(
                policy, robots, task_pos, task_payload, task_service, lam3, task_order)

        rewards, metrics = proxy_rewards(robots, task_pos, task_payload, task_service, assign, lam3, T_ref_s)

        pg, vf, ent = ppo_update_fast(policy, optimizer, robots, task_pos, task_payload, task_service,
                                      task_order, lam3, assign, logp0, rewards)

        row = [it, seed, n_robots_it, rewards.mean().item(),
               metrics["makespan_s"].mean().item(), metrics["max_soc_drop_pct"].mean().item(),
               metrics["cap_variance"].mean().item(), metrics["total_unserved"].mean().item(),
               pg, vf, ent]
        with open(log_path, "a") as f:
            f.write(",".join(f"{x:.5g}" if isinstance(x, float) else str(x) for x in row) + "\n")

        if (it + 1) % 10 == 0 or it + 1 == fhp.num_scenarios:
            print(f"[it {it + 1}/{fhp.num_scenarios} seed {seed} robots {n_robots_it}] "
                  f"reward {row[3]:.3f} | makespan {row[4]:.0f}s  max_soc {row[5]:.1f}%  "
                  f"cap_var {row[6]:.4f}  unserved {row[7]:.2f} | pg {pg:.3f} vf {vf:.3f} ent {ent:.2f}")

        if (it + 1) % fhp.save_every == 0 or it + 1 == fhp.num_scenarios:
            name = save_checkpoint(run_dir, it + 1, policy, optimizer)
            print(f"   saved {name}")


# ======================================================================
# Example: produce a task allocation vector for one scenario (no training)
# ======================================================================
@torch.no_grad()
def allocate(policy, robots, task_pos, task_payload, task_service, lam3, seed=0):
    """Deterministic (argmax) single-episode allocation. Returns the task allocation vector."""
    task_order = np.random.default_rng(seed).permutation(len(task_payload))
    lam3_t = torch.as_tensor([lam3], dtype=torch.float32, device=dev)
    M = len(robots["category"])
    N = len(task_payload)
    task_f = assignment_features(task_pos, task_payload, task_service)
    robot_f_static = robot_static_features(robots)
    cap = torch.as_tensor(robots["payload_kg"], dtype=torch.float32, device=dev)
    pay = torch.as_tensor(task_payload, dtype=torch.float32, device=dev)
    remaining = cap.clone().unsqueeze(0)

    assign = torch.full((1, N), -1, dtype=torch.long, device=dev)
    for j in task_order:
        cap_frac = (remaining / cap[None, :]).clamp(min=0.0)
        Ht, Hr, e = policy.encode(task_f, robot_f_static, cap_frac, lam3_t)
        logits = policy.step_logits(Ht, Hr, e, j)
        feasible = remaining >= pay[j]
        masked_logits = logits.masked_fill(~feasible, -1e9)
        if not feasible.any():
            continue
        choice = masked_logits.argmax(-1)
        assign[0, j] = choice
        remaining[0, choice] -= pay[j]

    return get_task_allocation_vector(assign[0])          # [N] : index = task id, value = robot id


if __name__ == "__main__":
    train()
