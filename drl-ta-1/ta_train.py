"""
train_assignment.py
--------------------
Top-layer (task ASSIGNMENT) training for heterogeneous multi-robot MRTA.

One-shot assignment network:
    - A single transformer forward pass encodes all tasks and all robots of
      a scenario into embeddings, then a [N_tasks, N_robots+1] compatibility
      logit matrix is produced in one shot (the "+1" column = "leave unassigned").
    - To turn those logits into a FEASIBLE assignment (robot payload capacity
      must not be exceeded), tasks are read out in a fixed order with a cheap
      sequential capacity mask applied to the *already-computed* logits (no
      re-encoding, no re-running the network -> still "one-shot" scoring).
    - This is the same idea used in attention-based combinatorial-optimisation
      policies (e.g. Kool et al.) adapted to set-to-set assignment.

Embedded lower layer (frozen):
    - For a given scenario + assignment, each robot's subset of assigned
      tasks is handed to the frozen single-drone PPO sequencer
      (FiLMPointerPolicy from train_ppo.py) which decides visiting order and
      speeds. Its resulting energy/time is read off via SingleDroneEnv +
      DroneEnergyModel. This makes the sequencer a pure black-box evaluator.

Objectives (top layer), combined with a 3D preference vector
w = [w_makespan, w_energy, w_variance] ~ Dirichlet, scalarised into one reward:
    1. makespan        = max_i T_i                     (max completion time)
    2. mean SOC-drop   = mean_i (E_i / usable_battery_i)   (energy, per robot
                          normalised so heterogeneous battery sizes compare fairly)
    3. payload variance = Var_i (assigned_payload_i / payload_capacity_i)
                          (capacity-normalised load balance)
w_makespan and w_energy are also translated into the sequencer's own local
2D preference lambda = [lambda_E, lambda_T] per robot (see `w_to_lambda`).

Files needed in the same folder: scenario_gen.py, drone_energy.py, train_ppo.py
A trained lower-layer checkpoint is expected at hp.lower_checkpoint (placeholder
path below -- update it once you have one; the script will warn and run with a
randomly-initialised sequencer if the file is missing, which is fine for a dry run).
"""

import os
import math
import glob
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Dirichlet

from scenario_gen import generate_scenario, DEPOTS, CATEGORIES
from drone_energy import DroneEnergyModel, RESERVE_SOC, SPEED_LEVELS
from train_ppo import FiLMPointerPolicy, hp as seq_hp


# ======================================================================
# HYPERPARAMETERS
# ======================================================================
@dataclass
class HP:
    run_id: str = "assign-run_002"
    save_root: str = "checkpoints_assign"
    save_every: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_seed: int = 1

    # ---- lower-layer (frozen) checkpoint --------------------------------
    lower_checkpoint: str = "ts3-run_004/ts3-run_004_it05000_ep10000.pt"   # <-- set this

    # ---- scenario sampling -----------------------------------------------
    robot_counts: tuple = (10, 15, 20)
    task_count_range: tuple = (30, 60)      # inclusive, sampled uniformly (covers 50..100)
    start_seed: int = 1
    num_iterations: int = 100
    scenarios_per_iter: int = 8             # batch of independent scenarios per PPO update

    # ---- PPO ---------------------------------------------------------------
    epochs_per_iter: int = 1
    lr: float = 3e-4
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # ---- objective weights / normalisation -------------------------------
    pref_alpha: float = 1.0          # Dirichlet(alpha,alpha,alpha) over [makespan, energy, variance]
    unassigned_penalty: float = 2.0  # reward penalty per task left unassigned
    makespan_norm_s: float = 3600.0  # rough normaliser so makespan is O(1) in the reward

    # ---- network -----------------------------------------------------------
    d_model: int = 128
    n_heads: int = 8
    n_enc_layers: int = 3
    ff_dim: int = 256


hp = HP()

TASK_DIM = 5     # same features as the lower layer: [x,y,z, payload/fleet_cap_ref, service/scale]
ROBOT_DIM = 8    # same as lower layer: 4 normalised specs + 4-way category one-hot


# ======================================================================
# Scenario batch generation (variable robots AND tasks per scenario)
# ======================================================================
def sample_scenario(seed):
    """One scenario with a random fleet size and random task count, mixed categories."""
    rng = np.random.default_rng(seed)
    n_robots = int(rng.choice(hp.robot_counts))
    n_tasks = int(rng.integers(hp.task_count_range[0], hp.task_count_range[1] + 1))
    robots, task_pos, task_payload, task_service, _, err = generate_scenario(
        n_robots, n_tasks, seed=seed, return_meta=True)   # random mixed categories
    return robots, task_pos, task_payload, task_service, err


def fleet_task_features(task_pos, task_payload, task_service, fleet_cap_ref):
    """Same shape convention as train_ppo.task_features, but payload is normalised
    by a fleet-level reference capacity (max single-robot capacity in CATEGORIES)
    since, at assignment time, we don't yet know which robot a task will go to."""
    f = np.column_stack([task_pos, task_payload / fleet_cap_ref,
                          task_service / seq_hp.service_scale])
    return torch.as_tensor(f, dtype=torch.float32, device=hp.device)


def all_robot_features(robots):
    """[M, ROBOT_DIM] features for every robot in the fleet (vectorised version
    of train_ppo.robot_features, which only handled a single robot)."""
    M = len(robots["category"])
    spec = np.stack([
        robots["payload_kg"] / max(CATEGORIES["payload_kg"]),
        robots["battery_wh"] / max(CATEGORIES["battery_wh"]),
        robots["flight_min"] / max(CATEGORIES["flight_min"]),
        robots["speed_ms"] / max(CATEGORIES["speed_ms"]),
    ], axis=1)                                                       # [M,4]
    onehot = np.eye(4)[robots["category"]]                           # [M,4]
    feats = np.concatenate([spec, onehot], axis=1)                   # [M,8]
    return torch.as_tensor(feats, dtype=torch.float32, device=hp.device)


# ======================================================================
# Assignment network: one-shot task/robot encoder -> compatibility logits
# ======================================================================
class AssignmentPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        d = hp.d_model
        self.d = d
        self.task_mlp = nn.Sequential(nn.Linear(TASK_DIM, d), nn.ReLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=hp.n_heads, dim_feedforward=hp.ff_dim,
                                            dropout=0.0, batch_first=True)
        self.task_encoder = nn.TransformerEncoder(layer, num_layers=hp.n_enc_layers)
        self.robot_mlp = nn.Sequential(nn.Linear(ROBOT_DIM, d), nn.ReLU(), nn.Linear(d, d))
        # preference vector (3D) conditions both the compatibility scores and the critic
        self.pref_mlp = nn.Sequential(nn.Linear(3, d), nn.ReLU(), nn.Linear(d, d))
        self.Wq = nn.Linear(d, d)                 # task query
        self.Wk = nn.Linear(d, d, bias=False)     # robot key
        self.unassigned_logit = nn.Parameter(torch.zeros(1))   # learned bias for the "drop" column
        # critic: pooled task+robot+pref embedding -> scalar value
        self.critic = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, task_feats, robot_feats, w):
        """
        task_feats  [N, TASK_DIM]
        robot_feats [M, ROBOT_DIM]
        w           [3]  preference vector for this scenario
        Returns: logits [N, M+1] (last column = unassigned), value (scalar)
        """
        N = task_feats.shape[0]
        Ht = self.task_encoder(self.task_mlp(task_feats)[None])[0]        # [N,d]
        Hr = self.robot_mlp(robot_feats)                                  # [M,d]
        e = self.pref_mlp(w[None])                                        # [1,d]

        q = self.Wq(Ht + e)                                                # [N,d]
        k = self.Wk(Hr)                                                    # [M,d]
        logits_robots = (q @ k.T) / math.sqrt(self.d)                      # [N,M]
        drop_col = self.unassigned_logit.expand(N, 1)
        logits = torch.cat([logits_robots, drop_col], dim=1)               # [N,M+1]

        pooled = torch.cat([Ht.mean(0), Hr.mean(0), e[0]], dim=0)          # [3d]
        value = self.critic(pooled).squeeze(-1)
        return logits, value


# ======================================================================
# Feasibility-masked sequential decode (reads the ONE-SHOT logits in order)
# ======================================================================
def decode_assignment(logits, task_payload, robot_capacity, deterministic=False,
                       chosen=None):
    """
    logits         [N, M+1] (already computed by one network forward pass)
    task_payload   [N] kg
    robot_capacity [M] kg (remaining capacity, consumed as tasks are assigned)
    chosen         [N] long, optional -> replay these choices instead of sampling
                   (used by PPO to re-evaluate log-probs under updated weights)

    Returns: chosen [N] long (index in 0..M, M = "unassigned"),
             logp (sum over tasks), entropy (sum over tasks)
    """
    N, Mp1 = logits.shape
    M = Mp1 - 1
    remaining = robot_capacity.clone()
    out = torch.zeros(N, dtype=torch.long, device=logits.device)
    logp_sum = torch.zeros((), device=logits.device)
    ent_sum = torch.zeros((), device=logits.device)

    for i in range(N):
        mask = torch.cat([remaining < task_payload[i], torch.tensor([False], device=logits.device)])
        masked_logits = logits[i].masked_fill(mask, -1e9)
        dist = Categorical(logits=masked_logits)
        if chosen is not None:
            a = chosen[i]
        else:
            a = masked_logits.argmax() if deterministic else dist.sample()
        logp_sum = logp_sum + dist.log_prob(a)
        ent_sum = ent_sum + dist.entropy()
        out[i] = a
        if a < M:
            remaining[a] = remaining[a] - task_payload[i]
    return out, logp_sum, ent_sum


# ======================================================================
# Objective evaluation: run the FROZEN lower-layer sequencer for ALL robots
# of the fleet TOGETHER, as one batch (instead of looping robot-by-robot).
# Each robot's assigned tasks are padded to a common length Nmax and the
# padding is masked out of selection -- this is exactly what SingleDroneEnv's
# B dimension already does for parallel rollouts of ONE robot; here B = M
# robots of the SAME scenario instead, each with its own heterogeneous
# physics (via model.leg_cost's per-row `ids`) and its own task subset.
# ======================================================================
def w_to_lambda(w):
    """Top-layer [w_makespan, w_energy, w_variance] -> sequencer's local [lam_E, lam_T]."""
    s = w[0] + w[1] + 1e-8
    lam_T = (w[0] / s).item()
    lam_E = (w[1] / s).item()
    return torch.tensor([lam_E, lam_T], dtype=torch.float32, device=hp.device)


class BatchedFleetEnv:
    """Like SingleDroneEnv, but each batch row is a DIFFERENT robot with its
    own category/physics and its own (padded) task subset, run in parallel."""

    def __init__(self, robots, assign, task_pos, task_payload, task_service, device):
        M = len(robots["category"])
        groups = [np.where(assign == r)[0] for r in range(M)]
        Nmax = max(1, max(len(g) for g in groups))
        f32 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)

        self.M, self.Nmax, self.dev = M, Nmax, device
        self.model = DroneEnergyModel(robots, device=device)     # fleet-wide energy model
        self.ids = torch.arange(M, device=device)                # row i = robot i, directly

        depots = DEPOTS[robots["depot_id"]]                      # [M,3]
        pos = np.repeat(depots[:, None, :], Nmax, axis=1).astype(float)   # pad -> own depot (0 extra dist)
        pay = np.zeros((M, Nmax)); srv = np.zeros((M, Nmax)); valid = np.zeros((M, Nmax), dtype=bool)
        for r, g in enumerate(groups):
            n = len(g)
            if n:
                pos[r, :n], pay[r, :n], srv[r, :n], valid[r, :n] = task_pos[g], task_payload[g], task_service[g], True

        self.pos, self.pay, self.srv = f32(pos), f32(pay), f32(srv)
        self.valid = torch.as_tensor(valid, device=device)
        self.depot = f32(depots)
        self.cap = f32(robots["payload_kg"])
        self.battery = f32(robots["battery_wh"])
        self.usable = (1.0 - RESERVE_SOC) * self.battery
        self.vmax = f32(robots["speed_ms"])
        self.fracs = torch.as_tensor(SPEED_LEVELS, device=device)
        self.v_ret = self.fracs[-1] * self.vmax                  # conservative feasibility proxy (fastest)

    def reset(self):
        self.cur = self.depot.clone()
        self.load = (self.pay * self.valid).sum(1)
        self.E_used = torch.zeros(self.M, device=self.dev)
        self.T_used = torch.zeros(self.M, device=self.dev)
        self.visited = ~self.valid                               # padding starts "visited" -> never picked
        self.active = self.valid.any(1)                          # robots with 0 tasks start inactive
        self.update_feasibility()

    def update_feasibility(self):
        M, N, S = self.M, self.Nmax, len(SPEED_LEVELS)
        ids = self.ids[:, None, None].expand(M, N, S)
        p_from = self.cur[:, None, None, :].expand(M, N, S, 3)
        p_to = self.pos[:, :, None, :].expand(M, N, S, 3)
        load = self.load[:, None, None].expand(M, N, S)
        srv = self.srv[:, :, None].expand(M, N, S)
        speed = (self.fracs[None, :] * self.vmax[:, None])[:, None, :].expand(M, N, S)

        E_go, _ = self.model.leg_cost(ids, p_from, p_to, load, srv, speed)
        load_after = (self.load[:, None] - self.pay).clamp(min=0.0)             # [M,N]
        ids_back = self.ids[:, None].expand(M, N)
        E_back, _ = self.model.leg_cost(ids_back, self.pos, self.depot[:, None, :].expand(M, N, 3),
                                         load_after, torch.zeros(M, N, device=self.dev),
                                         self.v_ret[:, None].expand(M, N))

        budget = (self.usable - self.E_used)[:, None, None]
        self.ok = (E_go + E_back[..., None] <= budget) & (~self.visited)[:, :, None]

        dist = (self.pos - self.cur[:, None, :]).norm(dim=-1)                   # [M,N]
        margin = budget.view(M, 1) - E_go[..., -1] - E_back
        self.speed_feat = torch.stack([dist, margin / self.battery[:, None]], dim=-1)

    def observe(self):
        return torch.cat([self.cur, (self.load / self.cap)[:, None],
                           ((self.usable - self.E_used) / self.battery)[:, None],
                           self.visited.float().mean(1, keepdim=True)], dim=1)   # [M,6]

    def step(self, task, speed_frac, x_ret):
        M, S = self.M, len(SPEED_LEVELS)
        act, rows = self.active, torch.arange(M, device=self.dev)
        p_to = self.pos[rows, task]
        v = speed_frac * self.vmax
        E, T = self.model.leg_cost(self.ids, self.cur, p_to, self.load, self.srv[rows, task], v)
        E, T = E * act, T * act

        self.cur = torch.where(act[:, None], p_to, self.cur)
        self.load = (self.load - self.pay[rows, task] * act).clamp(min=0.0)
        self.visited[rows[act], task[act]] = True
        self.E_used = self.E_used + E
        self.T_used = self.T_used + T
        self.update_feasibility()
        finished = act & (self.visited.all(1) | ~self.ok.flatten(1).any(1))

        budget = self.usable - self.E_used                                      # feasible return-speed grid
        speed_grid = self.fracs[None, :] * self.vmax[:, None]
        ids_g = self.ids[:, None].expand(M, S)
        pf_g, pt_g = self.cur[:, None, :].expand(M, S, 3), self.depot[:, None, :].expand(M, S, 3)
        load_g, zero_g = self.load[:, None].expand(M, S), torch.zeros(M, S, device=self.dev)
        E_grid, _ = self.model.leg_cost(ids_g, pf_g, pt_g, load_g, zero_g, speed_grid)
        ok_grid = E_grid <= budget[:, None]
        idx = torch.arange(S, device=self.dev).expand_as(ok_grid)
        lo = torch.where(ok_grid, idx, torch.full_like(idx, S)).min(-1).values.clamp(max=S - 1)
        hi = torch.where(ok_grid, idx, torch.full_like(idx, -1)).max(-1).values.clamp(min=0)
        no_feas = hi < lo
        lo, hi = torch.where(no_feas, torch.zeros_like(lo), lo), torch.where(no_feas, torch.zeros_like(hi), hi)
        frac_lo, frac_hi = self.fracs[lo], self.fracs[hi]
        ret_frac = frac_lo + (frac_hi - frac_lo) * x_ret
        E_ret, T_ret = self.model.leg_cost(self.ids, self.cur, self.depot, self.load,
                                            torch.zeros(M, device=self.dev), ret_frac * self.vmax)
        E_ret, T_ret = E_ret * finished, T_ret * finished
        self.E_used, self.T_used = self.E_used + E_ret, self.T_used + T_ret
        self.active = act & ~finished


def fleet_task_features_batched(pos, pay, srv, cap):
    """[M,Nmax,TASK_DIM]; payload normalised by EACH ROW'S OWN robot capacity,
    matching exactly how the lower layer was trained (train_ppo.task_features)."""
    payload_n = pay / cap[:, None].clamp(min=1e-6)
    service_n = srv / seq_hp.service_scale
    return torch.cat([pos, payload_n[..., None], service_n[..., None]], dim=-1)


def batched_policy_act(seq_policy, task_f, robot_f, state, lam, ok, speed_feat,
                        load_n, t_used_n, depot, pad_mask):
    """Same math as FiLMPointerPolicy.act (deterministic), but task_f/robot_f/state
    are all PER-ROW (one scenario each row = one robot), encoded in a single
    batched transformer pass instead of one call per robot."""
    p = seq_policy
    e = p.pref_mlp(lam)
    H = p.task_encoder(p.task_mlp(task_f), src_key_padding_mask=pad_mask)        # [M,N,d]
    H1 = p.film1(H, e)

    s = p.state_enc(torch.cat([robot_f, state], dim=-1))
    task_mask = ~ok.any(-1)
    free = (~task_mask).float().unsqueeze(-1)
    pooled = (H1 * free).sum(1) / free.sum(1).clamp(min=1.0)
    c = p.film2(p.ctx_mlp(torch.cat([pooled, s], dim=-1)), e)
    q = p.film3(p.Wq(c), e)
    k = p.Wk(H1)
    logits = seq_hp.logit_clip * torch.tanh((q.unsqueeze(1) * k).sum(-1) / math.sqrt(p.d))
    logits = logits.masked_fill(task_mask, -1e9)
    task = logits.argmax(-1)                                                      # deterministic

    rows = torch.arange(task.shape[0], device=task.device)
    ok_bs, h_i, sf_i, tf_i = ok[rows, task], H1[rows, task], speed_feat[rows, task], task_f[rows, task]

    speed_in = torch.cat([q, h_i, sf_i, load_n[:, None], t_used_n[:, None]], dim=-1)
    a_raw, b_raw = p.speed_head(speed_in).chunk(2, dim=-1)
    alpha, beta_ = F.softplus(a_raw.squeeze(-1)) + 1.0, F.softplus(b_raw.squeeze(-1)) + 1.0
    x = alpha / (alpha + beta_)
    frac_lo, frac_hi = FiLMPointerPolicy._feasible_range(ok_bs, p.speed_fracs)
    speed_frac = frac_lo + (frac_hi - frac_lo) * x

    load_after_n = (load_n - tf_i[:, 3]).clamp(min=0.0)
    dist_to_depot = (tf_i[:, :3] - depot).norm(dim=-1)
    ret_in = torch.cat([c, dist_to_depot[:, None], load_after_n[:, None], t_used_n[:, None]], dim=-1)
    ar_raw, br_raw = p.return_speed_head(ret_in).chunk(2, dim=-1)
    alpha_r, beta_r = F.softplus(ar_raw.squeeze(-1)) + 1.0, F.softplus(br_raw.squeeze(-1)) + 1.0
    x_ret = alpha_r / (alpha_r + beta_r)

    return task, speed_frac, x_ret


@torch.no_grad()
def evaluate_assignment(seq_policy, robots, task_pos, task_payload, task_service, assign, w):
    """Runs the frozen sequencer for every robot in the fleet AT ONCE (batched
    over robots), instead of looping robot-by-robot. Returns the three raw
    objective values plus the count of unassigned tasks."""
    env = BatchedFleetEnv(robots, assign, task_pos, task_payload, task_service, hp.device)
    env.reset()
    M = env.M
    lam = w_to_lambda(w)[None, :].expand(M, 2)
    task_f = fleet_task_features_batched(env.pos, env.pay, env.srv, env.cap)
    robot_f = all_robot_features(robots)
    pad_mask = ~env.valid                                     # padding only, constant across steps

    for _ in range(env.Nmax):
        if not env.active.any():
            break
        state = env.observe()
        task, speed_frac, x_ret = batched_policy_act(
            seq_policy, task_f, robot_f, state, lam, env.ok, env.speed_feat,
            env.load / env.cap, env.T_used / 3600.0, env.depot, pad_mask)
        env.step(task, speed_frac, x_ret)

    load_frac = (env.pay * env.valid).sum(1) / env.cap.clamp(min=1e-6)
    makespan = env.T_used.max().item() if M > 0 else 0.0
    mean_soc_drop = (env.E_used / env.usable.clamp(min=1e-6)).mean().item()
    payload_var = load_frac.var(unbiased=False).item()
    n_unassigned = int((assign == M).sum())
    return makespan, mean_soc_drop, payload_var, n_unassigned


def scalarize(makespan, mean_soc_drop, payload_var, n_unassigned, w):
    """Weighted-sum scalarisation (swap for Chebyshev later if desired)."""
    r = -(w[0].item() * (makespan / hp.makespan_norm_s)
          + w[1].item() * mean_soc_drop
          + w[2].item() * payload_var)
    r -= hp.unassigned_penalty * n_unassigned
    return r


# ======================================================================
# One scenario rollout (one "macro-step" of the assignment policy)
# ======================================================================
def run_scenario(assign_policy, seq_policy, seed):
    robots, task_pos, task_payload, task_service, err = sample_scenario(seed)
    if err is not None or len(task_payload) == 0:
        return None

    fleet_cap_ref = max(CATEGORIES["payload_kg"])
    task_f = fleet_task_features(task_pos, task_payload, task_service, fleet_cap_ref)
    robot_f = all_robot_features(robots)
    w = Dirichlet(torch.full((3,), hp.pref_alpha)).sample().to(hp.device)

    logits, value = assign_policy(task_f, robot_f, w)
    capacity = torch.as_tensor(robots["payload_kg"], dtype=torch.float32, device=hp.device)
    payload_t = torch.as_tensor(task_payload, dtype=torch.float32, device=hp.device)
    assign, logp, ent = decode_assignment(logits, payload_t, capacity)

    makespan, mean_soc, pvar, n_un = evaluate_assignment(
        seq_policy, robots, task_pos, task_payload, task_service, assign.cpu().numpy(), w)
    reward = scalarize(makespan, mean_soc, pvar, n_un, w)

    return {
        "task_f": task_f, "robot_f": robot_f, "w": w, "capacity": capacity,
        "payload": payload_t, "assign": assign, "logp": logp.detach(),
        "value": value.detach(), "reward": torch.tensor(reward, device=hp.device),
        "entropy": ent.detach(),
        "log_row": [seed, len(robots["category"]), len(task_payload),
                    makespan, mean_soc, pvar, n_un, reward],
    }


# ======================================================================
# PPO update over a batch of independent scenarios
# ======================================================================
def ppo_update(assign_policy, optimizer, batch):
    rewards = torch.stack([b["reward"] for b in batch])
    values = torch.stack([b["value"] for b in batch])
    adv = rewards - values
    if len(batch) > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = []
    for epoch in range(hp.epochs_per_iter):
        pg_loss = 0.0
        v_loss = 0.0
        ent_loss = 0.0
        optimizer.zero_grad()
        for b, a in zip(batch, adv):
            logits, value = assign_policy(b["task_f"], b["robot_f"], b["w"])
            _, logp_new, ent_new = decode_assignment(
                logits, b["payload"], b["capacity"], chosen=b["assign"])
            ratio = torch.exp(logp_new - b["logp"])
            pg = -torch.min(ratio * a.detach(),
                             torch.clamp(ratio, 1 - hp.clip_eps, 1 + hp.clip_eps) * a.detach())
            vf = (value - b["reward"]).pow(2)
            pg_loss = pg_loss + pg
            v_loss = v_loss + vf
            ent_loss = ent_loss + ent_new
        n = len(batch)
        loss = (pg_loss + hp.value_coef * v_loss) / n - hp.entropy_coef * (ent_loss / n)
        loss.backward()
        nn.utils.clip_grad_norm_(assign_policy.parameters(), hp.max_grad_norm)
        optimizer.step()
        stats.append([(pg_loss / n).item(), (v_loss / n).item(), (ent_loss / n).item()])
    return np.mean(stats, axis=0)


# ======================================================================
# Frozen lower-layer loading
# ======================================================================
def load_frozen_sequencer():
    policy = FiLMPointerPolicy().to(hp.device)
    if os.path.exists(hp.lower_checkpoint):
        ckpt = torch.load(hp.lower_checkpoint, map_location=hp.device, weights_only=True)
        policy.load_state_dict(ckpt["model"])
        print(f"Loaded frozen sequencer from {hp.lower_checkpoint}")
    else:
        print(f"[WARNING] lower_checkpoint not found at '{hp.lower_checkpoint}'. "
              f"Running with a randomly-initialised sequencer (dry run only -- "
              f"set hp.lower_checkpoint before real training).")
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


# ======================================================================
# Training loop
# ======================================================================
def train():
    torch.manual_seed(hp.torch_seed)
    run_dir = os.path.join(hp.save_root, hp.run_id)
    os.makedirs(run_dir, exist_ok=True)

    seq_policy = load_frozen_sequencer()
    assign_policy = AssignmentPolicy().to(hp.device)
    optimizer = torch.optim.Adam(assign_policy.parameters(), lr=hp.lr)

    log_path = os.path.join(run_dir, "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("iter,seed,n_robots,n_tasks,makespan,mean_soc_drop,payload_var,"
                    "n_unassigned,reward,pg_loss,v_loss,entropy\n")

    seed = hp.start_seed
    for it in range(hp.num_iterations):
        batch = []
        while len(batch) < hp.scenarios_per_iter:
            res = run_scenario(assign_policy, seq_policy, seed)
            seed += 1
            if res is not None:
                batch.append(res)

        pg, vf, ent = ppo_update(assign_policy, optimizer, batch)

        with open(log_path, "a") as f:
            for b in batch:
                row = [it] + b["log_row"] + [pg, vf, ent]
                f.write(",".join(f"{x:.5g}" if isinstance(x, float) else str(x) for x in row) + "\n")

        if (it + 1) % 10 == 0:
            mean_r = np.mean([b["reward"].item() for b in batch])
            print(f"[it {it + 1}/{hp.num_iterations}] mean reward {mean_r:.3f} | "
                  f"pg {pg:.3f} vf {vf:.3f} ent {ent:.3f}")

        if (it + 1) % hp.save_every == 0 or it + 1 == hp.num_iterations:
            path = os.path.join(run_dir, f"{hp.run_id}_it{it + 1:05d}.pt")
            torch.save({"model": assign_policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "hparams": asdict(hp)}, path)
            print(f"   saved {path}")


if __name__ == "__main__":
    train()