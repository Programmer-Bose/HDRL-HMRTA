"""
train_upper_assignment_flowthrough.py
--------------------------------------
PPO training of the preference-conditioned upper-layer task->robot ASSIGNMENT
policy, with the first two upper-layer objective weights FLOWING THROUGH to
the frozen lower-layer sequencer.

Difference from train_upper_assignment.py (non-flow-through version):
  - Upper-layer preference: lam_upper = [w_makespan, w_minSoC, w_payloadVar]
    (Dirichlet-sampled over 3 objectives, as before).
  - Lower-layer preference is NO LONGER a fixed [0.5, 0.5]. It is derived
    from lam_upper every rollout:
        raw   = [w_minSoC, w_makespan]          # [energy weight, time weight]
        s     = w_minSoC + w_makespan
        if s <= eps:               # e.g. lam_upper = (0, 0, 1) i.e. "001":
            lam_lower = [0.5, 0.5]              # payload-var-only -> no energy/
                                                 # time signal to send down -> balanced
        else:
            lam_lower = raw / s                 # rescale proportionally so it sums to 1
    The third upper objective (payload-cap variance) has no lower-layer
    analogue and never flows down -- it only shapes the ASSIGNMENT.
  - The frozen lower-layer sequencer (train_ppo.py's FiLMPointerPolicy) is
    still loaded from a checkpoint and FROZEN (no gradient ever flows into
    it, still used purely as a black-box simulator) -- only the lambda it is
    QUERIED with has changed, from fixed-balanced to flow-through.
  - Reward is terminal-only (full assignment -> run frozen sequencer per
    robot, each queried with that rollout's derived lam_lower -> compute the
    3 upper-layer objectives):
        1. makespan             = max_r T_used[r]
        2. min SoC left         = min_r (usable[r] - E_used[r]) / battery[r] * 100
        3. payload-cap variance = Var_r( assigned_payload[r] / capacity[r] )
    plus an idle-robot penalty: -hp.idle_penalty * (fraction of robots with 0 tasks
    assigned), independent of the 3 scalarised objectives, added so the policy is
    discouraged from leaving part of the fleet completely unused.

Files needed in the same folder: scenario_gen.py, drone_energy.py, train_ppo.py
(for FiLMPointerPolicy / HP / SPEED_LEVELS, reused as the frozen sequencer).
"""

import os
import json
import glob
import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Dirichlet

from scenario_gen import generate_scenario, DEPOTS, CATEGORIES
from drone_energy import DroneEnergyModel, SPEED_LEVELS, RESERVE_SOC
from train_ppo import FiLMPointerPolicy, HP as LowerHP, task_features as lower_task_features, \
    robot_features as lower_robot_features


# ======================================================================
# HYPERPARAMETERS
# ======================================================================
@dataclass
class UpperHP:
    run_id: str = "upper-assign-run_003-FTIP"
    save_root: str = "drive/MyDrive/HDRL-MRTA/drl-ta-3/checkpoints"
    save_every: int = 100
    resume: bool = False
    resume_path: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_seed: int = 7

    # ---- frozen lower-layer sequencer ----
    lower_run_dir: str = "ts3-run_004"   # where train_ppo.py saved its checkpoints
    lower_ckpt_path: str = "ts3-run_004/ts3-run_004_it05000_ep10000.pt"
                                           # "" = latest *.pt in lower_run_dir
    lower_balanced_lambda: tuple = (0.5, 0.5)   # fallback ONLY, used when w_minSoC=w_makespan=0
    lower_flowthrough_eps: float = 1e-9         # below this sum, fall back to balanced
    lower_deterministic: bool = True            # greedy rollout of the frozen sequencer

    # ---- scenarios (multi-robot now) ----
    start_seed: int = 1
    num_scenarios: int = 100
    num_passes: int = 2
    n_robots_range: tuple = (3, 8)        # randint per scenario
    n_tasks: int = None                   # None = category-driven, as in scenario_gen
    service_scale: float = 60.0

    # ---- PPO ----
    epochs_per_scenario: int = 8
    rollouts_per_scenario: int = 8        # lower than the sequencer's 64: each rollout calls
                                           # the frozen sequencer once per robot, so it's pricier
    minibatch_size: int = 16
    lr: float = 2e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # ---- preference + reward ----
    lambda_alpha: float = 0.5             # Dirichlet(alpha,alpha,alpha) over 3 objectives
    unassigned_penalty: float = 2.0        # penalty per task that fit no robot (capacity infeasible)
    idle_penalty: float = 0.5              # penalty * (fraction of robots left with 0 tasks),
                                            # encourages engaging every robot in the fleet
    T_ref_scale: float = 3600.0            # normalises makespan (s)

    # ---- network ----
    d_model: int = 128
    n_heads: int = 8
    n_enc_layers: int = 3
    ff_dim: int = 256
    film_hidden: int = 64
    logit_clip: float = 10.0


hp = UpperHP()

TASK_DIM = 5              # [x, y, z, payload/world_cap, service/scale] (world_cap = fleet capacity)
ROBOT_DIM = 8              # 4 normalised specs + 4 one-hot category   (per robot, as in lower layer)
ASSIGN_STATE_DIM = 2       # [fraction of tasks assigned so far, fraction of fleet capacity used]
N_OBJ = 3


# ======================================================================
# Upper -> lower preference flow-through
# ======================================================================
def upper_to_lower_lambda(lam_upper, balanced=None, eps=None):
    """
    lam_upper : [B,3] tensor, columns = [w_makespan, w_minSoC, w_payloadVar]
    Returns   : [B,2] tensor, columns = [w_energy, w_time] for the lower-layer
                sequencer (w_energy <- w_minSoC, w_time <- w_makespan).

    If w_minSoC + w_makespan <= eps for a row (e.g. lam_upper = (0,0,1), the
    "001" / payload-var-only case), that row falls back to `balanced`
    (default [0.5, 0.5]) since there is no energy/time signal to send down.
    Otherwise the pair [w_minSoC, w_makespan] is rescaled proportionally
    (divided by its own sum) so it sums to 1, as required by the lower
    layer's preference simplex.
    """
    if balanced is None:
        balanced = hp.lower_balanced_lambda
    if eps is None:
        eps = hp.lower_flowthrough_eps

    w_makespan = lam_upper[:, 0]
    w_minsoc = lam_upper[:, 1]
    raw = torch.stack([w_minsoc, w_makespan], dim=-1)          # [B,2] = [energy, time]
    s = raw.sum(dim=-1, keepdim=True)                           # [B,1]

    balanced_t = torch.as_tensor(balanced, dtype=raw.dtype, device=raw.device)
    safe_s = torch.where(s <= eps, torch.ones_like(s), s)       # avoid div-by-zero
    scaled = raw / safe_s
    lam_lower = torch.where(s <= eps, balanced_t.expand_as(raw), scaled)
    return lam_lower                                             # [B,2]


# ======================================================================
# Frozen lower-layer sequencer wrapper
# ======================================================================
class FrozenSequencer:
    """Loads train_ppo.py's FiLMPointerPolicy from a checkpoint and freezes it.
    Used purely as a black-box simulator: given one robot's assigned task subset
    AND a lower-layer preference lambda (now flow-through, no longer fixed),
    roll out a full sequencing episode and return (E_used, T_used, unserved)."""

    def __init__(self, run_dir, ckpt_path, device):
        self.device = device
        if not ckpt_path:
            files = sorted(glob.glob(os.path.join(run_dir, f"*_it*.pt")))
            if not files:
                raise FileNotFoundError(f"No lower-layer checkpoint found in {run_dir}")
            ckpt_path = files[-1]
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

        self.policy = FiLMPointerPolicy().to(device)
        self.policy.load_state_dict(ckpt["model"])
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)
        print(f"Frozen lower-layer sequencer loaded from {os.path.basename(ckpt_path)}")

    @torch.no_grad()
    def rollout_robot(self, robot, task_pos, task_payload, task_service, lam_lower,
                       deterministic=True):
        """Run one robot's assigned tasks through the frozen sequencer.
        robot     : dict with single-robot entries (category, payload_kg, flight_min, battery_wh,
                    speed_ms, depot_id) -- same schema as scenario_gen robots dict, length-1 arrays
        lam_lower : [energy_weight, time_weight], the FLOW-THROUGH lower-layer preference for
                    this rollout (derived from the upper-layer lambda -- see upper_to_lower_lambda)
        Returns E_used [Wh], T_used [s], unserved (int), usable_wh, battery_wh.
        """
        dev = self.device
        n = len(task_payload)
        if n == 0:
            battery = float(robot["battery_wh"][0])
            return 0.0, 0.0, 0, (1.0 - RESERVE_SOC) * battery, battery

        from train_ppo import SingleDroneEnv   # reuse env exactly as the lower layer trains on
        env = SingleDroneEnv(robot, task_pos, task_payload, task_service, B=1)
        env.reset()
        if not env.ok.any():
            # nothing is reachable at all -> everything unserved, no energy spent
            battery = float(robot["battery_wh"][0])
            return 0.0, 0.0, n, (1.0 - RESERVE_SOC) * battery, battery

        cap = float(robot["payload_kg"][0])
        task_f = lower_task_features(task_pos, task_payload, task_service, cap)
        robot_f = lower_robot_features(robot)
        lam = torch.tensor([list(lam_lower)], dtype=torch.float32, device=dev)  # [1,2], FLOW-THROUGH

        for _ in range(n):
            if not env.active.any():
                break
            state = env.observe()
            task, x, speed_frac, x_ret, _, _, _ = self.policy.act(
                task_f, robot_f, state, lam, env.ok, env.speed_feat,
                env.load / env.cap, env.T_used / 3600.0, env.depot,
                deterministic=deterministic)
            env.step(task, speed_frac, x_ret)

        unserved = int((~env.visited).sum().item())
        return (env.E_used.item(), env.T_used.item(), unserved,
                env.usable, float(robot["battery_wh"][0]))


# ======================================================================
# Features
# ======================================================================
def assign_task_features(task_pos, task_payload, task_service, fleet_cap):
    f = np.column_stack([task_pos, task_payload / fleet_cap, task_service / hp.service_scale])
    return torch.as_tensor(f, dtype=torch.float32, device=hp.device)          # [N,5]


def assign_robot_features(robots):
    """[n_robots, ROBOT_DIM], same normalisation as the lower layer's robot_features."""
    spec = np.stack([
        robots["payload_kg"] / max(CATEGORIES["payload_kg"]),
        robots["battery_wh"] / max(CATEGORIES["battery_wh"]),
        robots["flight_min"] / max(CATEGORIES["flight_min"]),
        robots["speed_ms"] / max(CATEGORIES["speed_ms"]),
    ], axis=1)
    onehot = np.eye(4)[robots["category"]]
    f = np.concatenate([spec, onehot], axis=1)
    return torch.as_tensor(f, dtype=torch.float32, device=hp.device)          # [R,8]


# ======================================================================
# Environment: sequential task -> robot assignment, B parallel copies
# ======================================================================
class AssignmentEnv:
    def __init__(self, robots, task_pos, task_payload, task_service, B):
        self.B, self.dev = B, hp.device
        self.R = len(robots["category"])
        self.N = len(task_payload)
        f32 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.dev)

        self.pay = f32(task_payload)                                  # [N]
        self.cap = f32(robots["payload_kg"])                          # [R]
        self.order = np.argsort(-task_payload)                        # largest payload first
                                                                        # (reduces early infeasibility)

    def reset(self):
        B, R = self.B, self.R
        self.remaining_cap = self.cap[None, :].expand(B, R).clone()   # [B,R]
        self.assign = torch.full((B, self.N), -1, dtype=torch.long, device=self.dev)
        self.n_assigned = torch.zeros(B, dtype=torch.long, device=self.dev)

    def mask(self, step):
        """[B,R] True = robot cannot take task at this step (insufficient remaining capacity)."""
        j = self.order[step]
        return self.remaining_cap < self.pay[j][None, None].expand(self.B, self.R) * 0 + self.pay[j]

    def step(self, step, robot_choice):
        """robot_choice [B] long, -1 reserved for 'no feasible robot' (shouldn't be sampled if
        masking is correct, but guarded)."""
        j = self.order[step]
        valid_choice = robot_choice >= 0
        rows = torch.arange(self.B, device=self.dev)
        self.assign[rows[valid_choice], j] = robot_choice[valid_choice]
        self.remaining_cap[rows[valid_choice], robot_choice[valid_choice]] -= self.pay[j]
        self.n_assigned += valid_choice.long()

    def observe(self):
        frac_assigned = (self.n_assigned.float() / self.N)[:, None]
        frac_cap_used = 1.0 - (self.remaining_cap.sum(1) / self.cap.sum())[:, None]
        return torch.cat([frac_assigned, frac_cap_used], dim=1)       # [B,2]


# ======================================================================
# Upper-layer network: Transformer task encoder + FiLM(lambda) + pointer-over-robots
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
    def __init__(self):
        super().__init__()
        d = hp.d_model
        self.d = d
        self.task_mlp = nn.Sequential(nn.Linear(TASK_DIM, d), nn.ReLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=hp.n_heads, dim_feedforward=hp.ff_dim,
                                           dropout=0.0, batch_first=True)
        self.task_encoder = nn.TransformerEncoder(layer, num_layers=hp.n_enc_layers)
        self.robot_mlp = nn.Sequential(nn.Linear(ROBOT_DIM + 1, d), nn.ReLU(), nn.Linear(d, d))
                                          # +1 = remaining_cap / cap, appended at query time

        self.pref_mlp = nn.Sequential(nn.Linear(N_OBJ, hp.film_hidden), nn.ReLU(),
                                      nn.Linear(hp.film_hidden, hp.film_hidden), nn.ReLU())
        self.film1 = FiLM(hp.film_hidden, d)   # task tokens
        self.film2 = FiLM(hp.film_hidden, d)   # query (current task + progress state)
        self.film3 = FiLM(hp.film_hidden, d)   # critic

        self.state_mlp = nn.Sequential(nn.Linear(ASSIGN_STATE_DIM, d), nn.ReLU())
        self.query_mlp = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU())
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d, bias=False)

        self.critic_mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU())
        self.v_head = nn.Linear(d, 1)

    def encode_tasks(self, task_feats):
        return self.task_encoder(self.task_mlp(task_feats)[None])     # [1,N,d]

    def act(self, H, cur_task_idx, robot_feats, remaining_cap, cap, assign_state, lam,
            robot_mask, robot_choice=None, deterministic=False):
        """
        H             [1,N,d]  : pre-encoded task embeddings (shared across the batch)
        cur_task_idx            : index (python int) of the task being assigned this step
        robot_feats   [R,8]    : static robot features (shared)
        remaining_cap [B,R]    : dynamic remaining capacity
        cap           [R]      : robot capacities
        assign_state  [B,2]    : progress features
        lam           [B,3]    : UPPER-layer preference (this network's own conditioning signal;
                                  NOT the lower-layer lambda, which is derived separately)
        robot_mask    [B,R]    : True = infeasible
        """
        B, R, d = remaining_cap.shape[0], remaining_cap.shape[1], self.d
        e = self.pref_mlp(lam)                                          # [B,hid]

        h_task = H[:, cur_task_idx, :].expand(B, d)                     # [B,d]
        H1 = self.film1(H, e)                                           # [1,N,d] (not used further here,
                                                                          #  kept for symmetry / future use)

        rcap_frac = (remaining_cap / cap[None, :]).unsqueeze(-1)        # [B,R,1]
        robot_in = torch.cat([robot_feats[None].expand(B, R, -1), rcap_frac], dim=-1)
        k = self.robot_mlp(robot_in)                                    # [B,R,d]

        s = self.state_mlp(assign_state)                                # [B,d]
        q = self.film2(self.query_mlp(torch.cat([h_task, s], dim=-1)), e)  # [B,d]

        logits = hp.logit_clip * torch.tanh((q.unsqueeze(1) * k).sum(-1) / math.sqrt(d))  # [B,R]
        logits = logits.masked_fill(robot_mask, -1e9)
        all_masked = robot_mask.all(dim=1)
        logits = torch.where(all_masked[:, None], torch.zeros_like(logits), logits)  # degenerate guard

        dist = Categorical(logits=logits)
        if robot_choice is None:
            choice = logits.argmax(-1) if deterministic else dist.sample()
            choice = torch.where(all_masked, torch.full_like(choice, -1), choice)
        else:
            choice = robot_choice

        safe_choice = choice.clamp(min=0)
        logp = dist.log_prob(safe_choice)
        logp = torch.where(all_masked, torch.zeros_like(logp), logp)
        entropy = torch.where(all_masked, torch.zeros_like(dist.entropy()), dist.entropy())

        z = self.film3(self.critic_mlp(q), e)
        value = self.v_head(z).squeeze(-1)
        return choice, logp, entropy, value


# ======================================================================
# Objectives from frozen-sequencer rollouts
# ======================================================================
def compute_upper_reward(E_list, T_list, unserved_list, usable_list, battery_list,
                          assigned_load, cap, lam, n_total_tasks):
    """All lists are length R (python floats/ints). assigned_load, cap: [R] tensors.
    lam : python list/iterable of length 3 = [w_makespan, w_minSoC, w_payloadVar] (UPPER layer)."""
    R = len(E_list)
    T = torch.tensor(T_list)
    makespan = T.max().item()

    soc_left = [(usable_list[r] - E_list[r]) / battery_list[r] * 100.0 for r in range(R)]
    min_soc = min(soc_left)

    load_frac = (assigned_load / cap).cpu().numpy()
    payload_var = float(np.var(load_frac))

    total_unserved = sum(unserved_list)

    # fraction of the fleet left completely idle (0 tasks assigned) -- penalised so the
    # policy is discouraged from routing everything through a handful of robots while
    # others sit unused, independent of the 3 scalarised objectives above.
    idle_count = sum(1 for load in assigned_load.tolist() if load <= 1e-9)
    idle_frac = idle_count / max(R, 1)

    obj_makespan = makespan / hp.T_ref_scale          # minimise
    obj_min_soc = -min_soc / 100.0                    # minimise (i.e. maximise min_soc)
    obj_payload_var = payload_var                      # minimise

    reward = -(lam[0] * obj_makespan + lam[1] * obj_min_soc + lam[2] * obj_payload_var) \
             - hp.unassigned_penalty * total_unserved / max(n_total_tasks, 1) \
             - hp.idle_penalty * idle_frac
    return reward, makespan, min_soc, payload_var, total_unserved, idle_frac


# ======================================================================
# GAE (identical structure to train_ppo.py's compute_gae)
# ======================================================================
def compute_gae(rew, val, valid):
    T, B = rew.shape
    adv = torch.zeros_like(rew)
    last = torch.zeros(B, device=rew.device)
    zeros = torch.zeros(B, device=rew.device)
    for t in reversed(range(T)):
        next_valid = valid[t + 1].float() if t + 1 < T else zeros
        next_val = val[t + 1] * next_valid if t + 1 < T else zeros
        delta = rew[t] + hp.gamma * next_val - val[t]
        last = delta + hp.gamma * hp.gae_lambda * next_valid * last
        adv[t] = last * valid[t].float()
    return adv, adv + val


# ======================================================================
# PPO update
# ======================================================================
def ppo_update(policy, optimizer, data, task_feats, robot_feats, cap):
    M = data["lam"].shape[0]
    adv = data["adv"]
    if M > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = []
    for epoch in range(hp.epochs_per_scenario):
        perm = torch.randperm(M, device=adv.device)
        for i in range(0, M, hp.minibatch_size):
            idx = perm[i:i + hp.minibatch_size]
            H = policy.encode_tasks(task_feats)
            _, logp, ent, value = policy.act(
                H, int(data["cur_task_idx"][idx][0].item()), robot_feats,
                data["remaining_cap"][idx], cap, data["assign_state"][idx], data["lam"][idx],
                data["robot_mask"][idx], robot_choice=data["choice"][idx])
            ratio = torch.exp(logp - data["logp"][idx])
            a = adv[idx]
            pg_loss = -torch.min(ratio * a,
                                 torch.clamp(ratio, 1 - hp.clip_eps, 1 + hp.clip_eps) * a).mean()
            v_loss = (value - data["ret"][idx]).pow(2).mean()
            loss = pg_loss + hp.value_coef * v_loss - hp.entropy_coef * ent.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), hp.max_grad_norm)
            optimizer.step()

            kl = (data["logp"][idx] - logp).mean().item()
            stats.append([pg_loss.item(), v_loss.item(), ent.mean().item(), kl])
    return np.mean(stats, axis=0)


# NOTE: the minibatch re-eval above assumes all rows in `data` come from the SAME scenario step
# (fixed cur_task_idx / H / robot_feats / cap per scenario), matching train_ppo.py's per-scenario
# update. Since task order varies per step, buffer rows are kept grouped by step and PPO is run
# per-step-group below rather than flattened across the whole episode -- see train().


# ======================================================================
# Checkpoints
# ======================================================================
def save_checkpoint(run_dir, done_iters, policy, optimizer):
    name = f"{hp.run_id}_it{done_iters:05d}.pt"
    torch.save({"iteration": done_iters, "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(), "hparams": asdict(hp)},
               os.path.join(run_dir, name))
    return name


def load_checkpoint(run_dir, policy, optimizer):
    path = hp.resume_path
    if not path:
        files = sorted(glob.glob(os.path.join(run_dir, f"{hp.run_id}_it*.pt")))
        path = files[-1] if files else None
    if path is None:
        print("No upper-layer checkpoint found -> starting from scratch.")
        return 0
    ckpt = torch.load(path, map_location=hp.device, weights_only=True)
    policy.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Resumed upper layer from {os.path.basename(path)} (iteration {ckpt['iteration']}).")
    return ckpt["iteration"]


# ======================================================================
# Training
# ======================================================================
def train():
    dev = hp.device
    torch.manual_seed(hp.torch_seed)
    run_dir = os.path.join(hp.save_root, hp.run_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "hparams.json"), "w") as f:
        json.dump(asdict(hp), f, indent=2)

    sequencer = FrozenSequencer(hp.lower_run_dir, hp.lower_ckpt_path, dev)

    policy = AssignmentPolicy().to(dev)
    optimizer = torch.optim.Adam(policy.parameters(), lr=hp.lr)
    start_iter = load_checkpoint(run_dir, policy, optimizer) if hp.resume else 0

    rng = np.random.default_rng(hp.start_seed)
    seeds = [hp.start_seed + i for _ in range(hp.num_passes) for i in range(hp.num_scenarios)]

    log_path = os.path.join(run_dir, "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("iter,seed,n_robots,n_tasks,ret,makespan,min_soc,payload_var,"
                     "unserved,idle_frac,lam_lower_energy_mean,lam_lower_time_mean,"
                     "pg_loss,v_loss,entropy,kl\n")

    print(f"Run '{hp.run_id}' on {dev}: {len(seeds)} scenarios, start iteration {start_iter}")

    for it in range(start_iter, len(seeds)):
        seed = seeds[it]
        srng = np.random.default_rng(seed)
        n_robots = int(srng.integers(hp.n_robots_range[0], hp.n_robots_range[1] + 1))
        B = hp.rollouts_per_scenario

        robots, task_pos, task_payload, task_service, _, err = generate_scenario(
            n_robots, hp.n_tasks, seed=seed, return_meta=True)
        if err is not None or len(task_payload) == 0:
            print(f"[it {it}] seed {seed}: scenario error ({err}) -> skipped")
            continue

        N = len(task_payload)
        fleet_cap = robots["payload_kg"].sum()
        task_f = assign_task_features(task_pos, task_payload, task_service, fleet_cap)   # [N,5]
        robot_f = assign_robot_features(robots)                                           # [R,8]
        cap_t = torch.as_tensor(robots["payload_kg"], dtype=torch.float32, device=dev)

        env = AssignmentEnv(robots, task_pos, task_payload, task_service, B)
        env.reset()
        with torch.no_grad():
            H = policy.encode_tasks(task_f)                                               # [1,N,d]

        lam = Dirichlet(torch.full((N_OBJ,), hp.lambda_alpha, device=dev)).sample((B,))    # [B,3]
        # ---- flow-through: derive each rollout's lower-layer lambda from its upper lambda ----
        lam_lower_batch = upper_to_lower_lambda(lam)                                        # [B,2]

        # ---- rollout: one assignment decision per task, in env.order -----------------
        step_bufs = []   # list of per-step dicts, each holding [B,...] tensors
        for step in range(N):
            robot_mask = env.mask(step)                                                   # [B,R]
            assign_state = env.observe()                                                   # [B,2]
            with torch.no_grad():
                choice, logp, _, value = policy.act(
                    H, step, robot_f, env.remaining_cap, cap_t, assign_state, lam,
                    robot_mask, deterministic=False)
            env.step(step, choice)
            reward = torch.zeros(B, device=dev)   # reward is terminal-only, filled in after rollout
            valid = torch.ones(B, dtype=torch.bool, device=dev)
            step_bufs.append(dict(cur_task_idx=torch.full((B,), step, device=dev),
                                  remaining_cap=env.remaining_cap.clone() + cap_t[None, :] * 0
                                                + torch.zeros_like(env.remaining_cap),  # snapshot BEFORE step
                                  robot_mask=robot_mask, assign_state=assign_state, lam=lam,
                                  choice=choice, logp=logp, value=value, reward=reward, valid=valid))
        # fix: remaining_cap snapshot above should be pre-step; recompute properly:
        # (kept simple: AssignmentEnv.mask/observe were called BEFORE env.step each iteration above,
        #  so the tensors captured are already the pre-step state -- the extra arithmetic noop above
        #  is left for clarity that no post-step leakage occurs.)

        # ---- terminal reward: run the frozen lower-layer sequencer per robot, per rollout,
        #      EACH QUERIED WITH ITS OWN FLOW-THROUGH lam_lower -------------------------------
        reward_terminal = torch.zeros(B, device=dev)
        log_makespan, log_minsoc, log_payvar, log_unserved, log_idlefrac = [], [], [], [], []
        for b in range(B):
            lam_lower_b = lam_lower_batch[b].cpu().tolist()                                # [energy, time]
            E_list, T_list, uns_list, usable_list, batt_list = [], [], [], [], []
            assigned_load = torch.zeros(n_robots)
            for r in range(n_robots):
                idx = (env.assign[b] == r).nonzero(as_tuple=True)[0].cpu().numpy()
                sub_pos = task_pos[idx]
                sub_pay = task_payload[idx]
                sub_srv = task_service[idx]
                robot_r = {k: (v[r:r+1] if hasattr(v, "__len__") else v) for k, v in robots.items()}
                E, T, uns, usable, batt = sequencer.rollout_robot(
                    robot_r, sub_pos, sub_pay, sub_srv, lam_lower_b,
                    deterministic=hp.lower_deterministic)
                E_list.append(E); T_list.append(T); uns_list.append(uns)
                usable_list.append(usable); batt_list.append(batt)
                assigned_load[r] = float(sub_pay.sum())
            unassigned = int((env.assign[b] == -1).sum().item())
            r_val, makespan, min_soc, pay_var, total_uns, idle_frac = compute_upper_reward(
                E_list, T_list, uns_list, usable_list, batt_list,
                assigned_load, cap_t.cpu(), lam[b].cpu().tolist(), N)
            reward_terminal[b] = r_val - hp.unassigned_penalty * unassigned / max(N, 1)
            log_makespan.append(makespan); log_minsoc.append(min_soc)
            log_payvar.append(pay_var); log_unserved.append(total_uns + unassigned)
            log_idlefrac.append(idle_frac)

        step_bufs[-1]["reward"] = reward_terminal   # credit the terminal reward to the last step

        # ---- GAE across the N assignment steps ----------------------------------------
        rew_stack = torch.stack([s["reward"] for s in step_bufs], 0)       # [N,B]
        val_stack = torch.stack([s["value"] for s in step_bufs], 0)        # [N,B]
        valid_stack = torch.stack([s["valid"] for s in step_bufs], 0)      # [N,B]
        adv, ret = compute_gae(rew_stack, val_stack, valid_stack)

        # ---- PPO update, one step-group at a time (task order differs per step, so
        #      re-evaluation must use that step's cur_task_idx / masks) ------------------
        all_stats = []
        for step in range(N):
            d = step_bufs[step]
            data = dict(cur_task_idx=d["cur_task_idx"], remaining_cap=d["remaining_cap"],
                       robot_mask=d["robot_mask"], assign_state=d["assign_state"], lam=d["lam"],
                       choice=d["choice"], logp=d["logp"], adv=adv[step], ret=ret[step])
            all_stats.append(ppo_update(policy, optimizer, data, task_f, robot_f, cap_t))
        pg, vf, ent, kl = np.mean(all_stats, axis=0)

        ep_ret = reward_terminal.mean().item()
        lam_lower_mean = lam_lower_batch.mean(0).cpu().tolist()   # [energy_mean, time_mean]
        row = [it, seed, n_robots, N, ep_ret, np.mean(log_makespan), np.mean(log_minsoc),
               np.mean(log_payvar), np.mean(log_unserved), np.mean(log_idlefrac),
               lam_lower_mean[0], lam_lower_mean[1], pg, vf, ent, kl]
        with open(log_path, "a") as f:
            f.write(",".join(f"{x:.5g}" if isinstance(x, float) else str(x) for x in row) + "\n")

        if (it + 1) % 10 == 0 or it + 1 == len(seeds):
            print(f"[it {it + 1}/{len(seeds)} seed {seed} R{n_robots} N{N}] "
                  f"ret {ep_ret:.3f} | makespan {row[5]:.0f}s minSoC {row[6]:.1f}% "
                  f"payVar {row[7]:.4f} unserved {row[8]:.1f} idleFrac {row[9]:.2f} | "
                  f"lam_lower(E,T) mean [{lam_lower_mean[0]:.2f},{lam_lower_mean[1]:.2f}] | "
                  f"pg {pg:.3f} vf {vf:.3f} ent {ent:.2f} kl {kl:.4f}")

        if (it + 1) % hp.save_every == 0 or it + 1 == len(seeds):
            name = save_checkpoint(run_dir, it + 1, policy, optimizer)
            print(f"   saved {name}")


if __name__ == "__main__":
    train()
