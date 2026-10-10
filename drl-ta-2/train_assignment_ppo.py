"""
task_assignment.py
-------------------
Upper layer: multi-robot TASK ASSIGNMENT, conditioned on a 3-component
preference vector lam3 = [w_makespan, w_max_soc_drop, w_cap_variance].

Design (as discussed):
  - The LOWER layer (train_ppo.py: FiLMPointerPolicy, single-drone pointer
    sequencer) is already trained and is FROZEN here. It is always queried
    with the balanced preference lam=[0.5, 0.5], regardless of lam3. This
    keeps the evaluation signal stable while the assignment network trains,
    so gradients reflect assignment quality, not a moving sequencer target.
  - The UPPER layer assigns every task to exactly one robot (sequentially,
    one task at a time, autoregressive), respecting payload capacity. No
    task can go to two robots (hard masking).
  - Once a full assignment is produced, each robot's task subset is handed
    to the frozen sequencer, which deterministically (greedy / argmax)
    produces an order + speeds. From that we read off, per robot:
        E_used (Wh), T_used (s), SOC_drop (%) = E_used / battery * 100
  - Episode reward (terminal, one value for the whole assignment):
        makespan        = max_i T_used_i                      (robot that returns last)
        max_soc_drop     = max_i SOC_drop_i                     (worst-hit battery)
        cap_variance      = Var_i( assigned_payload_i / capacity_i )
        reward = -( lam3[0]*makespan_norm
                    + lam3[1]*max_soc_drop_norm
                    + lam3[2]*cap_variance )
                 - unserved_penalty * (# tasks left unassigned / infeasible)

  - Training: REINFORCE with a learned value baseline (actor-critic),
    since the reward is terminal/episodic (one assignment -> one number),
    not per-step like the lower layer's PPO. Every assignment step shares
    the same episode return; a per-step critic reduces variance. This is
    straightforward to upgrade to full clipped-PPO (see note at bottom)
    if you want multiple epochs per scenario.

Files needed in the same folder (project files, already present):
    scenario_gen.py, drone_energy.py, train_ppo.py
"""

import os
import sys
import glob
import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# sys.path.insert(0, "/mnt/project")  # project files (scenario_gen, drone_energy, train_ppo)

from scenario_gen import generate_scenario, DEPOTS, CATEGORIES
from drone_energy import DroneEnergyModel, SPEED_LEVELS, RESERVE_SOC
import train_ppo as lower  # reuse FiLMPointerPolicy, SingleDroneEnv, hp, feature helpers


# ======================================================================
# HYPERPARAMETERS
# ======================================================================
@dataclass
class AHP:
    run_id: str = "assign-run_001"
    save_root: str = "checkpoints_assign"
    save_every: int = 20
    resume: bool = True
    resume_path: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_seed: int = 7

    lower_ckpt: str = r"ts3-run_004\ts3-run_004_it05000_ep10000.pt"          # path to the frozen lower-layer .pt checkpoint (required)

    start_seed: int = 1
    num_scenarios: int = 100
    n_robots_min: int = 4         # fleet size is RANDOMISED per scenario, in [min, max] (>1; else
    n_robots_max: int = 10        # no assignment decision exists), so the policy generalises across fleets
    n_tasks: int = None           # None = scenario_gen default per-category range

    lr: float = 3e-4
    gamma: float = 1.0             # episode is short (N tasks); 1.0 is fine with a terminal-only reward
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    epochs_per_scenario: int = 2
    rollouts_per_scenario: int = 16    # parallel assignment episodes (independent lam3 samples) per scenario

    unserved_penalty: float = 2.0

    d_model: int = 128
    n_heads: int = 8
    n_enc_layers: int = 2
    ff_dim: int = 256
    film_hidden: int = 64
    logit_clip: float = 10.0


ahp = AHP()

TASK_DIM = 5          # [x, y, z, payload (kg, raw), service/scale]
ROBOT_DIM = 8          # 4 normalised specs + 4 one-hot category
ROBOT_DYN_DIM = 1      # remaining-capacity fraction (dynamic, updates as tasks are assigned)
PREF_DIM = 3           # [w_makespan, w_max_soc_drop, w_cap_var]


def robot_features_batch(robots):
    """[M,8] normalised robot features, same formula as train_ppo.robot_features, vectorised."""
    cat = np.asarray(robots["category"])
    spec = np.stack([
        robots["payload_kg"] / max(CATEGORIES["payload_kg"]),
        robots["battery_wh"] / max(CATEGORIES["battery_wh"]),
        robots["flight_min"] / max(CATEGORIES["flight_min"]),
        robots["speed_ms"] / max(CATEGORIES["speed_ms"]),
    ], axis=1)
    onehot = np.eye(4)[cat]
    return torch.as_tensor(np.concatenate([spec, onehot], axis=1), dtype=torch.float32)


class BatchRolloutEnv:
    """
    Generalisation of train_ppo.SingleDroneEnv to a batch of E independent
    (episode, robot) elements that may belong to DIFFERENT robots and own
    DIFFERENT (variable-size) task subsets out of one shared N-task pool.

    This lets ALL (B assignment-episodes x M robots) sequencing rollouts for
    one scenario run as ONE vectorised pass instead of B*M separate calls --
    this is the batching fix for the slow evaluation step.

    ids          : [E] long, index into `robots` arrays (which real robot this element is)
    init_visited : [E,N] bool, True where task j is NOT owned by element e
                   (pre-masks every other robot's tasks so this element can
                   only ever pick from its own assigned subset)
    """

    def __init__(self, robots, task_pos, task_payload, task_service, ids, init_visited, device):
        self.ids = ids
        self.E, self.N = ids.shape[0], len(task_payload)
        self.dev = device
        f32 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)

        self.model = DroneEnergyModel(robots, device=device)         # supports heterogeneous ids natively
        self.pos = f32(task_pos)
        self.pay = f32(task_payload)
        self.srv = f32(task_service)

        depot_all = f32(DEPOTS)                                      # [4,3]
        depot_id = torch.as_tensor(np.asarray(robots["depot_id"]), device=device)[ids]
        self.depot = depot_all[depot_id]                             # [E,3]
        self.cap = f32(robots["payload_kg"])[ids]
        self.battery = f32(robots["battery_wh"])[ids]
        self.vmax = f32(robots["speed_ms"])[ids]
        self.fracs = f32(SPEED_LEVELS)
        self.v_ret = self.fracs[lower.hp.return_speed_level] * self.vmax         # conservative proxy, as in SingleDroneEnv
        self.usable = (1.0 - RESERVE_SOC) * self.battery
        self.init_visited = init_visited

    def reset(self):
        E, N = self.E, self.N
        self.cur = self.depot.clone()
        owned = (~self.init_visited).float()                                     # [E,N]
        self.load = (self.pay[None, :] * owned).sum(1)                           # initial payload on board
        self.E_used = torch.zeros(E, device=self.dev)
        self.T_used = torch.zeros(E, device=self.dev)
        self.visited = self.init_visited.clone()
        self.active = torch.ones(E, dtype=torch.bool, device=self.dev)
        self.update_feasibility()

    def update_feasibility(self):
        """Same logic/shapes as SingleDroneEnv.update_feasibility, generalised to self.ids."""
        E, N, S = self.E, self.N, len(SPEED_LEVELS)
        ids = self.ids[:, None, None].expand(E, N, S)
        p_from = self.cur[:, None, None, :]
        p_to = self.pos[None, :, None, :]
        load = self.load[:, None, None]
        speed = (self.fracs * self.vmax[:, None])[:, None, :]                    # [E,1,S]

        E_go, _ = self.model.leg_cost(ids, p_from, p_to, load, self.srv[None, :, None], speed)
        load_after = (load - self.pay[None, :, None]).clamp(min=0.0)
        ids_back = self.ids[:, None].expand(E, N)
        E_back, _ = self.model.leg_cost(
            ids_back, self.pos[None, :, :].expand(E, N, 3), self.depot[:, None, :].expand(E, N, 3),
            load_after.squeeze(-1), 0.0, self.v_ret[:, None].expand(E, N))

        budget = (self.usable - self.E_used)[:, None, None]
        self.ok = (E_go + E_back[..., None] <= budget) & (~self.visited)[:, :, None]

        dist = (p_to - p_from).squeeze(2).norm(dim=-1)
        margin = budget.view(E, 1) - E_go[..., -1] - E_back
        self.speed_feat = torch.stack([dist, margin / self.battery[:, None]], dim=-1)

    def observe(self):
        return torch.cat([
            self.cur, (self.load / self.cap)[:, None],
            ((self.usable - self.E_used) / self.battery)[:, None],
            self.visited.float().mean(1, keepdim=True),
        ], dim=1)

    def step(self, task, speed_frac, x_ret):
        E = self.E
        act = self.active
        rows = torch.arange(E, device=self.dev)

        p_to = self.pos[task]
        v = speed_frac * self.vmax
        Estep, Tstep = self.model.leg_cost(self.ids, self.cur, p_to, self.load, self.srv[task], v)
        Estep, Tstep = Estep * act, Tstep * act

        self.cur = torch.where(act[:, None], p_to, self.cur)
        self.load = (self.load - self.pay[task] * act).clamp(min=0.0)
        self.visited[rows[act], task[act]] = True
        self.E_used += Estep
        self.T_used += Tstep

        self.update_feasibility()
        finished = act & (self.visited.all(1) | ~self.ok.flatten(1).any(1))

        budget = self.usable - self.E_used
        S = len(SPEED_LEVELS)
        speed_grid = self.fracs[None, :] * self.vmax[:, None]                    # [E,S]
        Eg, _ = self.model.leg_cost(self.ids[:, None].expand(E, S), self.cur[:, None, :].expand(E, S, 3),
                                    self.depot[:, None, :].expand(E, S, 3), self.load[:, None].expand(E, S),
                                    torch.zeros(E, S, device=self.dev), speed_grid)
        ok_grid = Eg <= budget[:, None]
        idx = torch.arange(S, device=self.dev).expand_as(ok_grid)
        lo_idx = torch.where(ok_grid, idx, torch.full_like(idx, S)).min(-1).values.clamp(max=S - 1)
        hi_idx = torch.where(ok_grid, idx, torch.full_like(idx, -1)).max(-1).values.clamp(min=0)
        no_feas = hi_idx < lo_idx
        lo_idx = torch.where(no_feas, torch.zeros_like(lo_idx), lo_idx)
        hi_idx = torch.where(no_feas, torch.zeros_like(hi_idx), hi_idx)
        frac_lo, frac_hi = self.fracs[lo_idx], self.fracs[hi_idx]
        ret_frac = frac_lo + (frac_hi - frac_lo) * x_ret
        v_ret_used = ret_frac * self.vmax

        E_ret, T_ret = self.model.leg_cost(self.ids, self.cur, self.depot, self.load,
                                           torch.zeros(E, device=self.dev), v_ret_used)
        E_ret, T_ret = E_ret * finished, T_ret * finished
        self.E_used += E_ret
        self.T_used += T_ret
        self.active = act & ~finished


# ======================================================================
# Frozen lower-layer evaluator (BATCHED across all episodes x robots)
# ======================================================================
class FrozenSequencer:
    """Wraps a trained FiLMPointerPolicy; always queried at lam = [0.5, 0.5]."""

    def __init__(self, ckpt_path, device):
        self.device = device
        self.policy = lower.FiLMPointerPolicy().to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        self.policy.load_state_dict(ckpt["model"])
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def rollout_batch(self, robots, task_pos, task_payload, task_service, ids, init_visited):
        """
        ONE vectorised deterministic rollout for E = (B episodes x M robots) elements at once.
        ids, init_visited : as in BatchRolloutEnv.
        Returns E_used [E], T_used [E], n_unserved [E] (all torch tensors on self.device).
        """
        dev = self.device
        env = BatchRolloutEnv(robots, task_pos, task_payload, task_service, ids, init_visited, dev)
        env.reset()

        N = env.N
        pay_t = torch.as_tensor(task_payload, dtype=torch.float32, device=dev)
        srv_t = torch.as_tensor(task_service, dtype=torch.float32, device=dev)
        pos_t = torch.as_tensor(task_pos, dtype=torch.float32, device=dev)

        # The lower policy's task encoder expects ONE shared [N,5] tensor whose payload column
        # is normalised by THIS robot's own capacity (exactly how it was trained -- see
        # train_ppo.task_features). Capacity only takes 4 distinct values (one per category), so
        # instead of approximating with a fixed reference cap, split the batch into (at most 4)
        # per-category groups, run the encoder once per group with the EXACT normalisation that
        # group's robots were trained under, then scatter the chosen actions back together and
        # advance the full batch with one env.step. Still a handful of vectorised calls per
        # step, not E individual ones -- the expensive part (per-task, per-robot rollout) stays batched.
        cat_of_id = torch.as_tensor(np.asarray(robots["category"]), device=dev)[ids]   # [E]
        groups = [torch.where(cat_of_id == c)[0] for c in cat_of_id.unique()]
        task_f_by_cat = {}
        for g in groups:
            c = int(cat_of_id[g[0]])
            cap_c = float(np.asarray(robots["payload_kg"])[np.asarray(robots["category"]) == c][0])
            task_f_by_cat[c] = torch.cat([pos_t, (pay_t / cap_c)[:, None],
                                          (srv_t / lower.hp.service_scale)[:, None]], dim=1)

        robot_f_all = robot_features_batch(robots).to(dev)            # [M,8]
        robot_f = robot_f_all[ids]                                    # [E,8]
        lam_balanced = torch.tensor([[0.5, 0.5]], device=dev).expand(env.E, 2)

        for _ in range(N):
            if not env.active.any():
                break
            state = env.observe()
            task = torch.zeros(env.E, dtype=torch.long, device=dev)
            speed_frac = torch.zeros(env.E, device=dev)
            x_ret = torch.zeros(env.E, device=dev)
            for g in groups:
                c = int(cat_of_id[g[0]])
                t_g, _, sf_g, xr_g, _, _, _ = self.policy.act(
                    task_f_by_cat[c], robot_f[g], state[g], lam_balanced[g], env.ok[g],
                    env.speed_feat[g], (env.load / env.cap)[g], (env.T_used / 3600.0)[g],
                    env.depot[g], deterministic=True)
                task[g], speed_frac[g], x_ret[g] = t_g, sf_g, xr_g
            env.step(task, speed_frac, x_ret)

        n_unserved = (~env.visited).sum(1).float()
        return env.E_used, env.T_used, n_unserved


def evaluate_assignments_batched(sequencer, robots, task_pos, task_payload, task_service, assign):
    """
    Batched evaluation of B assignments at once (ALL B*M robot rollouts in a single
    vectorised pass -- replaces the old per-(b,robot) Python loop).

    assign : torch.long [B,N], robot index per task (-1 = unassigned).
    Returns dict of torch tensors, each [B] (or [B,M] where noted):
        makespan_s, max_soc_drop_pct, cap_variance, total_unserved
    """
    dev = sequencer.device
    B, N = assign.shape
    M = len(robots["category"])

    ids = torch.arange(M, device=dev).repeat(B)                       # [B*M], robot id per element
    assign_exp = assign.repeat_interleave(M, dim=0)                   # [B*M, N]
    robot_rows = torch.arange(M, device=dev).repeat(B)[:, None]       # [B*M,1]
    init_visited = assign_exp != robot_rows                           # [B*M,N] True where NOT this element's task

    E_used, T_used, n_unserved = sequencer.rollout_batch(
        robots, task_pos, task_payload, task_service, ids, init_visited)

    E_used = E_used.view(B, M)
    T_used = T_used.view(B, M)
    n_unserved = n_unserved.view(B, M).sum(1) + (assign < 0).sum(1).float()   # + never-assigned tasks

    battery = torch.as_tensor(robots["battery_wh"], dtype=torch.float32, device=dev)
    cap = torch.as_tensor(robots["payload_kg"], dtype=torch.float32, device=dev)
    pay = torch.as_tensor(task_payload, dtype=torch.float32, device=dev)

    soc_drop = E_used / battery[None, :] * 100.0                      # [B,M]
    one_hot = F.one_hot(assign.clamp(min=0), num_classes=M).float() * (assign >= 0).float()[..., None]
    assigned_payload = torch.einsum("bn,bnm->bm", pay[None, :].expand(B, N), one_hot)  # [B,M]
    payload_frac = assigned_payload / cap[None, :]

    makespan = T_used.max(1).values
    max_soc_drop = soc_drop.max(1).values
    cap_variance = payload_frac.var(1, unbiased=False)

    return {
        "makespan_s": makespan, "max_soc_drop_pct": max_soc_drop,
        "cap_variance": cap_variance, "total_unserved": n_unserved,
    }


# ======================================================================
# Upper-layer network: autoregressive, one robot choice per task
# ======================================================================
class FiLM(nn.Module):
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.lin = nn.Linear(cond_dim, 2 * feat_dim)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, h, cond):
        gamma, beta = self.lin(cond).chunk(2, dim=-1)
        if h.dim() == 3:
            gamma, beta = gamma.unsqueeze(1), beta.unsqueeze(1)
        return (1.0 + gamma) * h + beta


class AssignmentPolicy(nn.Module):
    """
    At step t (tasks processed in a fixed order, e.g. index order), choose a
    robot (or 'unassigned' if every robot is over capacity) for task t.
    Robots attend to each other (fleet-aware), tasks attend to each other,
    then a cross compatibility score [task_t, robot_i] is produced, FiLM-
    conditioned on the 3-d preference vector.
    """

    def __init__(self):
        super().__init__()
        d = ahp.d_model
        self.d = d

        self.task_mlp = nn.Sequential(nn.Linear(TASK_DIM, d), nn.ReLU(), nn.Linear(d, d))
        t_layer = nn.TransformerEncoderLayer(d_model=d, nhead=ahp.n_heads, dim_feedforward=ahp.ff_dim,
                                             dropout=0.0, batch_first=True)
        self.task_encoder = nn.TransformerEncoder(t_layer, num_layers=ahp.n_enc_layers)

        self.robot_mlp = nn.Sequential(nn.Linear(ROBOT_DIM + ROBOT_DYN_DIM, d), nn.ReLU(), nn.Linear(d, d))
        r_layer = nn.TransformerEncoderLayer(d_model=d, nhead=ahp.n_heads, dim_feedforward=ahp.ff_dim,
                                             dropout=0.0, batch_first=True)
        self.robot_encoder = nn.TransformerEncoder(r_layer, num_layers=ahp.n_enc_layers)

        self.pref_mlp = nn.Sequential(nn.Linear(PREF_DIM, ahp.film_hidden), nn.ReLU(),
                                      nn.Linear(ahp.film_hidden, ahp.film_hidden), nn.ReLU())
        self.film_task = FiLM(ahp.film_hidden, d)
        self.film_robot = FiLM(ahp.film_hidden, d)
        self.film_query = FiLM(ahp.film_hidden, d)
        self.film_critic = FiLM(ahp.film_hidden, d)

        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d, bias=False)

        self.critic_mlp = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU())
        self.v_head = nn.Linear(d, 1)

    def encode(self, task_feats, robot_feats_static, robot_cap_frac, lam3):
        """
        task_feats          [N, TASK_DIM]
        robot_feats_static  [M, ROBOT_DIM]
        robot_cap_frac      [B, M]   remaining-capacity fraction, per episode (dynamic)
        lam3                [B, 3]
        Returns H_task [B,N,d] (broadcast), H_robot [B,M,d], e [B,hid]
        """
        B, M = robot_cap_frac.shape
        N = task_feats.shape[0]
        e = self.pref_mlp(lam3)

        Ht = self.task_encoder(self.task_mlp(task_feats)[None])          # [1,N,d]
        Ht = self.film_task(Ht.expand(B, N, -1), e)                      # [B,N,d]

        r_in = torch.cat([robot_feats_static.expand(B, M, -1), robot_cap_frac.unsqueeze(-1)], dim=-1)
        Hr = self.robot_encoder(self.robot_mlp(r_in))                    # [B,M,d]
        Hr = self.film_robot(Hr, e)
        return Ht, Hr, e

    def step_logits(self, Ht, Hr, e, task_idx):
        """Compatibility logits of task `task_idx` (scalar, same for whole batch since
        tasks are processed in a fixed shared order) against every robot. [B,M]"""
        B = Ht.shape[0]
        q = self.film_query(self.Wq(Ht[:, task_idx, :]), e)               # [B,d]
        k = self.Wk(Hr)                                                   # [B,M,d]
        logits = ahp.logit_clip * torch.tanh((q.unsqueeze(1) * k).sum(-1) / math.sqrt(self.d))
        return logits                                                     # [B,M]

    def value(self, Ht, Hr, e):
        pooled_t = Ht.mean(1)
        pooled_r = Hr.mean(1)
        z = self.film_critic(self.critic_mlp(torch.cat([pooled_t, pooled_r], dim=-1)), e)
        return self.v_head(z).squeeze(-1)                                 # [B]


# ======================================================================
# Assignment rollout (one scenario, B parallel lam3 samples)
# ======================================================================
def assignment_features(task_pos, task_payload, task_service):
    f = np.column_stack([task_pos, task_payload, task_service / lower.hp.service_scale])
    return torch.as_tensor(f, dtype=torch.float32, device=ahp.device)


def robot_static_features(robots):
    cat = robots["category"]
    spec = np.stack([
        robots["payload_kg"] / max(CATEGORIES["payload_kg"]),
        robots["battery_wh"] / max(CATEGORIES["battery_wh"]),
        robots["flight_min"] / max(CATEGORIES["flight_min"]),
        robots["speed_ms"] / max(CATEGORIES["speed_ms"]),
    ], axis=1)
    onehot = np.eye(4)[cat]
    return torch.as_tensor(np.concatenate([spec, onehot], axis=1), dtype=torch.float32, device=ahp.device)


def rollout_assignment(policy, robots, task_pos, task_payload, task_service, lam3, task_order=None):
    """
    Autoregressive assignment for B parallel episodes (different lam3 rows),
    SAME scenario. Capacity masking is tracked per (episode, robot).

    Returns: assign [B,N] long (-1 = unassigned), logp [B] (sum over steps),
             entropy [B] (sum), value [B] (from the step-0 state, used as baseline).
    """
    dev = ahp.device
    B = lam3.shape[0]
    M = len(robots["category"])
    N = len(task_payload)
    if task_order is None:
        task_order = np.arange(N)

    task_f = assignment_features(task_pos, task_payload, task_service)
    robot_f_static = robot_static_features(robots)
    cap = torch.as_tensor(robots["payload_kg"], dtype=torch.float32, device=dev)
    pay = torch.as_tensor(task_payload, dtype=torch.float32, device=dev)

    remaining = cap[None, :].expand(B, M).clone()                          # [B,M] kg left per robot
    assign = torch.full((B, N), -1, dtype=torch.long, device=dev)
    logp_sum = torch.zeros(B, device=dev)
    ent_sum = torch.zeros(B, device=dev)
    value0 = None

    for step, j in enumerate(task_order):
        cap_frac = (remaining / cap[None, :]).clamp(min=0.0)               # [B,M]
        Ht, Hr, e = policy.encode(task_f, robot_f_static, cap_frac, lam3)
        if step == 0:
            value0 = policy.value(Ht, Hr, e)

        logits = policy.step_logits(Ht, Hr, e, j)                          # [B,M]
        feasible = remaining >= pay[j]                                     # [B,M]
        masked_logits = logits.masked_fill(~feasible, -1e9)
        any_feasible = feasible.any(-1)                                    # [B]

        dist = Categorical(logits=masked_logits)
        choice = dist.sample()                                             # [B], garbage where infeasible
        logp = dist.log_prob(choice)
        ent = dist.entropy()

        logp_sum = logp_sum + logp * any_feasible
        ent_sum = ent_sum + ent * any_feasible

        assign[:, j] = torch.where(any_feasible, choice, torch.full_like(choice, -1))
        rows = torch.arange(B, device=dev)
        chosen_cap = remaining[rows, choice.clamp(min=0)]
        remaining[rows, choice.clamp(min=0)] = torch.where(
            any_feasible, chosen_cap - pay[j], chosen_cap)

    return assign, logp_sum, ent_sum, value0


# ======================================================================
# Reward (batched: all B episodes of one scenario at once)
# ======================================================================
def assignment_rewards_batched(sequencer, robots, task_pos, task_payload, task_service, assign,
                               lam3, T_ref_s, soc_ref_pct=100.0):
    """
    assign : torch.long [B,N].  lam3 : torch.float [B,3].
    Returns reward [B] (torch), metrics dict of [B] tensors.
    """
    metrics = evaluate_assignments_batched(sequencer, robots, task_pos, task_payload, task_service, assign)
    makespan_n = metrics["makespan_s"] / T_ref_s
    soc_n = metrics["max_soc_drop_pct"] / soc_ref_pct
    cap_var = metrics["cap_variance"]

    reward = -(lam3[:, 0] * makespan_n + lam3[:, 1] * soc_n + lam3[:, 2] * cap_var) \
             - ahp.unserved_penalty * metrics["total_unserved"]
    return reward, metrics


# ======================================================================
# PPO-style update (REINFORCE + value baseline; episode-terminal reward)
# ======================================================================
def ppo_update(policy, optimizer, robots, task_pos, task_payload, task_service, task_order,
               lam3, assign, old_logp, reward):
    """One (or a few) gradient epochs on a batch of B completed assignment episodes."""
    adv = reward - reward.mean()
    if reward.shape[0] > 1:
        adv = adv / (adv.std() + 1e-8)

    stats = []
    for _ in range(ahp.epochs_per_scenario):
        # re-evaluate logp/entropy/value under current params, replaying the SAME choices
        dev = ahp.device
        B = lam3.shape[0]
        M = len(robots["category"])
        N = len(task_payload)
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
            choice = assign[:, j].clamp(min=0)                               # replay stored choice
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
    name = f"{ahp.run_id}_it{it:05d}.pt"
    torch.save({"iteration": it, "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(), "hparams": asdict(ahp)},
               os.path.join(run_dir, name))
    return name


def load_checkpoint(run_dir, policy, optimizer):
    path = ahp.resume_path or (sorted(glob.glob(os.path.join(run_dir, f"{ahp.run_id}_it*.pt"))) or [None])[-1]
    if not path:
        print("No assignment checkpoint found -> starting from scratch.")
        return 0
    ckpt = torch.load(path, map_location=ahp.device, weights_only=True)
    policy.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Resumed assignment policy from {os.path.basename(path)} (iter {ckpt['iteration']}).")
    return ckpt["iteration"]


# ======================================================================
# Training
# ======================================================================
def train():
    assert ahp.lower_ckpt, "Set ahp.lower_ckpt to the frozen lower-layer sequencer checkpoint path."
    dev = ahp.device
    torch.manual_seed(ahp.torch_seed)
    run_dir = os.path.join(ahp.save_root, ahp.run_id)
    os.makedirs(run_dir, exist_ok=True)

    sequencer = FrozenSequencer(ahp.lower_ckpt, dev)
    policy = AssignmentPolicy().to(dev)
    optimizer = torch.optim.Adam(policy.parameters(), lr=ahp.lr)
    start_it = load_checkpoint(run_dir, policy, optimizer) if ahp.resume else 0

    log_path = os.path.join(run_dir, "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("iter,seed,reward_mean,makespan_s_mean,max_soc_mean,cap_var_mean,"
                    "unserved_mean,pg_loss,v_loss,entropy\n")

    for it in range(start_it, ahp.num_scenarios):
        seed = ahp.start_seed + it
        n_robots_it = int(np.random.default_rng(seed).integers(ahp.n_robots_min, ahp.n_robots_max + 1))
        robots, task_pos, task_payload, task_service, _, err = generate_scenario(
            n_robots_it, ahp.n_tasks, seed=seed, return_meta=True)
        if err is not None or len(task_payload) == 0:
            print(f"[it {it}] seed {seed}: scenario error ({err}) -> skipped")
            continue

        B = ahp.rollouts_per_scenario
        # sample preference vectors on the simplex (Dirichlet) so all three
        # objectives get explored, including corner cases (pure makespan, etc.)
        lam3 = torch.distributions.Dirichlet(torch.ones(3, device=dev)).sample((B,))

        T_ref_s = float(np.max(robots["flight_min"])) * 60.0   # normalisation reference
        task_order = np.random.permutation(len(task_payload))  # fixed shared order within this episode

        with torch.no_grad():
            assign, logp0, _, _ = rollout_assignment(
                policy, robots, task_pos, task_payload, task_service, lam3, task_order)

        # single batched call: evaluates all B assignments (B*M robot rollouts) at once
        rewards, metrics = assignment_rewards_batched(
            sequencer, robots, task_pos, task_payload, task_service, assign, lam3, T_ref_s)

        pg, vf, ent = ppo_update(policy, optimizer, robots, task_pos, task_payload, task_service,
                                 task_order, lam3, assign, logp0, rewards)

        row = [it, seed, rewards.mean().item(),
               metrics["makespan_s"].mean().item(),
               metrics["max_soc_drop_pct"].mean().item(),
               metrics["cap_variance"].mean().item(),
               metrics["total_unserved"].mean().item(),
               pg, vf, ent]
        with open(log_path, "a") as f:
            f.write(",".join(f"{x:.5g}" if isinstance(x, float) else str(x) for x in row) + "\n")

        if (it + 1) % 10 == 0 or it + 1 == ahp.num_scenarios:
            print(f"[it {it + 1}/{ahp.num_scenarios} seed {seed}] reward {row[2]:.3f} | "
                  f"makespan {row[3]:.0f}s  max_soc {row[4]:.1f}%  cap_var {row[5]:.4f}  "
                  f"unserved {row[6]:.2f} | pg {pg:.3f} vf {vf:.3f} ent {ent:.2f}")

        if (it + 1) % ahp.save_every == 0 or it + 1 == ahp.num_scenarios:
            name = save_checkpoint(run_dir, it + 1, policy, optimizer)
            print(f"   saved {name}")


if __name__ == "__main__":
    train()

# ----------------------------------------------------------------------
# Notes
# ----------------------------------------------------------------------
# 1. Upgrading to full PPO: store task_order/assign/lam3/old_logp/reward for
#    several scenarios in a replay buffer and shuffle minibatches across
#    scenarios before each epoch, instead of updating on one scenario at a
#    time -- reduces gradient noise from the terminal-only reward.
# 2. Caveat (1) from the design discussion: periodically re-run this script's
#    `evaluate_assignment` with a NON-frozen-balanced sequencer call (i.e.
#    FrozenSequencer loaded with lam != 0.5/0.5, one per lam3 corner) to check
#    the balanced-proxy reward still tracks true preference-conditioned
#    outcomes; fine-tune if the gap is large.
# 3. Caveat (2): `evaluate_assignment` always calls the sequencer at the
#    balanced lam, so SOC-drop and makespan numbers used for BOTH the reward
#    and the variance term come from the same (balanced) rollout -- never
#    mixed with a later preference-conditioned re-sequencing pass.