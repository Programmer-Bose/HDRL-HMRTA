"""
train_ppo.py
------------
PPO training of the preference-conditioned Transformer-FiLM pointer policy for
single-drone task sequencing (energy vs. makespan).

One training iteration = one scenario seed. train() runs, in order:
    1. generate the scenario                       (scenario_gen, 1 robot)
    2. build environment + energy model + reward scales   (drone_energy)
    3. sample preference vectors lambda
    4. roll out: network -> (task, speed) -> energy/time -> reward, step by step
    5. advantages (GAE)
    6. PPO update for `epochs_per_scenario` epochs
    7. log + checkpoint

Files needed in the same folder: scenario_gen.py, drone_energy.py
"""

import os
import json
import glob
import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Dirichlet

from scenario_gen import generate_scenario, DEPOTS, CATEGORIES
from drone_energy import DroneEnergyModel, SPEED_LEVELS, RESERVE_SOC


# ======================================================================
# HYPERPARAMETERS  (edit here only)
# ======================================================================
@dataclass
class HP:
    # ---- run / checkpoints ---------------------------------------------
    run_id: str = "run_004"            # checkpoints go to <save_root>/<run_id>/
    save_root: str = "checkpoints"
    save_every: int = 500               # save a checkpoint every N iterations (scenarios)
    resume: bool = True               # True -> continue this run_id from its latest checkpoint
    resume_path: str = ""              # optional explicit .pt file ("" = latest in the run folder)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_seed: int = 0

    # ---- scenarios -------------------------------------------------------
    start_seed: int = 123               # scenarios use seeds start_seed ... start_seed + num_scenarios - 1
    num_scenarios: int = 5000
    num_passes: int = 1                # how many times to sweep over the seed list
    n_robots: int = 1                  # task sequencing -> one drone
    n_tasks: int = None                  # None = random number of tasks per category (see scenario_gen.py)
    service_scale: float = 60.0        # [s] divides service time before it enters the network

    # ---- PPO -------------------------------------------------------------
    epochs_per_scenario: int = 5       # PPO epochs on each scenario's rollout batch
    rollouts_per_scenario: int = 64    # parallel episodes per scenario (each gets its own lambda)
    minibatch_size: int = 128
    lr: float = 2e-4
    gamma: float = 1.0                 # discount (episodes are short -> 1.0)
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # ---- preference + reward ------------------------------------------------
    lambda_alpha: float = 0.5          # Dirichlet(alpha, alpha) for [lambda_E, lambda_T]
    unserved_penalty: float = 1.0      # reward penalty per task the battery could not cover
    return_speed_level = 9        # speed level of the final return leg (index into SPEED_LEVELS)
    check_energy: bool = True          # cross-check step-wise energy against tour_energy()

    # ---- network -----------------------------------------------------------
    d_model: int = 128
    n_heads: int = 8
    n_enc_layers: int = 3
    ff_dim: int = 256
    film_hidden: int = 64
    logit_clip: float = 10.0


hp = HP()

TASK_DIM = 5                           # [x, y, z, payload/cap, service/scale]
ROBOT_DIM = 8                          # 4 normalised specs + 4 one-hot category
STATE_DIM = 6                          # [x, y, z, load/cap, usable battery left, fraction done]
N_SPEEDS = len(SPEED_LEVELS)


# ======================================================================
# Features
# ======================================================================
def task_features(task_pos, task_payload, task_service, cap):
    f = np.column_stack([task_pos, task_payload / cap, task_service / hp.service_scale])
    return torch.as_tensor(f, dtype=torch.float32, device=hp.device)          # [N,5]


def robot_features(robots):
    spec = np.array([
        robots["payload_kg"][0] / max(CATEGORIES["payload_kg"]),
        robots["battery_wh"][0] / max(CATEGORIES["battery_wh"]),
        robots["flight_min"][0] / max(CATEGORIES["flight_min"]),
        robots["speed_ms"][0] / max(CATEGORIES["speed_ms"]),
    ])
    onehot = np.eye(4)[int(robots["category"][0])]
    return torch.as_tensor(np.concatenate([spec, onehot]), dtype=torch.float32, device=hp.device)  # [8]


# ======================================================================
# Environment: B parallel copies of one scenario (one drone, N tasks)
# ======================================================================
class SingleDroneEnv:
    def __init__(self, robots, task_pos, task_payload, task_service, B):
        self.B, self.N, self.dev = B, len(task_pos), hp.device
        f32 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.dev)

        self.model = DroneEnergyModel(robots, device=self.dev)       # energy model (drone_energy.py)
        self.pos = f32(task_pos)                                     # [N,3]
        self.pay = f32(task_payload)                                 # [N]
        self.srv = f32(task_service)                                 # [N]
        self.depot = f32(DEPOTS[int(robots["depot_id"][0])])         # [3]
        self.cap = float(robots["payload_kg"][0])
        self.battery = float(robots["battery_wh"][0])
        self.usable = (1.0 - RESERVE_SOC) * self.battery             # [Wh] energy we may use
        self.vmax = float(robots["speed_ms"][0])
        self.fracs = f32(SPEED_LEVELS)                               # [S]
        self.v_ret = self.fracs[hp.return_speed_level] * self.vmax   # speed of the final return leg

    # ------------------------------------------------------------------
    def reset(self):
        B, N = self.B, self.N
        self.cur = self.depot.expand(B, 3).clone()                               # current position
        self.load = torch.full((B,), float(self.pay.sum()), device=self.dev)     # payload on board [kg]
        self.E_used = torch.zeros(B, device=self.dev)                            # energy used [Wh]
        self.T_used = torch.zeros(B, device=self.dev)                            # time used [s]
        self.visited = torch.zeros(B, N, dtype=torch.bool, device=self.dev)
        self.active = torch.ones(B, dtype=torch.bool, device=self.dev)
        self.update_feasibility()

    # ------------------------------------------------------------------
    def update_feasibility(self):
        """ok[b,j,s] = task j can be served at speed level s AND the drone can still fly home."""
        B, N, S = self.B, self.N, N_SPEEDS
        ids = torch.zeros(B, N, S, dtype=torch.long, device=self.dev)            # single robot -> id 0
        p_from = self.cur[:, None, None, :]                                      # [B,1,1,3]
        p_to = self.pos[None, :, None, :]                                        # [1,N,1,3]
        load = self.load[:, None, None]                                          # [B,1,1]
        speed = (self.fracs * self.vmax)[None, None, :]                          # [1,1,S]

        E_go, _ = self.model.leg_cost(ids, p_from, p_to, load, self.srv[None, :, None], speed)
        load_after = (load - self.pay[None, :, None]).clamp(min=0.0)             # [B,N,1]
        E_back, _ = self.model.leg_cost(ids, p_to, self.depot, load_after, 0.0, self.v_ret)

        budget = (self.usable - self.E_used)[:, None, None]
        self.ok = (E_go + E_back <= budget) & (~self.visited)[:, :, None]        # [B,N,S]

    # ------------------------------------------------------------------
    def observe(self):
        """Dynamic state fed to the network, [B,6]."""
        return torch.cat([
            self.cur,
            (self.load / self.cap)[:, None],
            ((self.usable - self.E_used) / self.battery)[:, None],
            self.visited.float().mean(1, keepdim=True),
        ], dim=1)

    # ------------------------------------------------------------------
    def step(self, task, level):
        """
        Fly to `task` at speed level `level`, serve it (payload drops).
        If no task can be served afterwards, the drone returns to the depot and the episode ends.
        Returns E_step, T_step (include the return leg on the last step), unserved, finished.
        """
        B = self.B
        act = self.active
        rows = torch.arange(B, device=self.dev)
        ids = torch.zeros(B, dtype=torch.long, device=self.dev)

        p_to = self.pos[task]                                                    # [B,3]
        v = self.fracs[level] * self.vmax                                        # [B]
        E, T = self.model.leg_cost(ids, self.cur, p_to, self.load, self.srv[task], v)
        E, T = E * act, T * act                                                  # finished episodes do not move

        self.cur = torch.where(act[:, None], p_to, self.cur)
        self.load = (self.load - self.pay[task] * act).clamp(min=0.0)            # mass decreases after the drop
        self.visited[rows[act], task[act]] = True
        self.E_used = self.E_used + E
        self.T_used = self.T_used + T

        self.update_feasibility()
        finished = act & (self.visited.all(1) | ~self.ok.flatten(1).any(1))

        zero_service = torch.zeros(B, device=self.dev)
        E_ret, T_ret = self.model.leg_cost(ids, self.cur, self.depot.expand(B, 3),
                                           self.load, zero_service, self.v_ret)
        E_ret, T_ret = E_ret * finished, T_ret * finished
        self.E_used = self.E_used + E_ret
        self.T_used = self.T_used + T_ret
        unserved = (~self.visited).sum(1).float() * finished

        self.active = act & ~finished
        return E + E_ret, T + T_ret, unserved, finished


def reference_scales(env):
    """E_ref, T_ref from a nearest-neighbour tour at max speed; only used to normalise the reward."""
    pos, cur, left, order = env.pos.cpu().numpy(), env.depot.cpu().numpy(), list(range(env.N)), []
    while left:
        j = min(left, key=lambda k: np.linalg.norm(pos[k] - cur))
        order.append(j)
        left.remove(j)
        cur = pos[j]
    coords = torch.cat([env.depot[None], env.pos[order], env.depot[None]])[None]          # [1,N+2,3]
    out = env.model.tour_energy(torch.zeros(1, dtype=torch.long, device=env.dev),
                                coords, env.pay[order][None], env.srv[order][None])
    return out["energy_wh"].item(), out["time_s"].item()


def check_energy(env, tasks, levels):
    """Recompute fully-served episodes with tour_energy() and compare with the step-wise sum."""
    done = env.visited.all(1)
    if done.sum() == 0:
        return float("nan")
    order = tasks[:, done].T                                                              # [Bf,N]
    Bf = order.shape[0]
    ret = torch.full((Bf, 1), hp.return_speed_level, dtype=torch.long, device=env.dev)
    lv = torch.cat([levels[:, done].T, ret], dim=1)                                       # [Bf,N+1]
    dep = env.depot.expand(Bf, 1, 3)
    coords = torch.cat([dep, env.pos[order], dep], dim=1)                                 # [Bf,N+2,3]
    out = env.model.tour_energy(torch.zeros(Bf, dtype=torch.long, device=env.dev),
                                coords, env.pay[order], env.srv[order], speed_levels=lv)
    return (out["energy_wh"] - env.E_used[done]).abs().max().item()


# ======================================================================
# Neural network: Transformer task encoder + FiLM(lambda) + pointer actor + critic
# ======================================================================
class FiLM(nn.Module):
    """h' = (1 + gamma) * h + beta, with (gamma, beta) = Linear(preference embedding)."""

    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.lin = nn.Linear(cond_dim, 2 * feat_dim)
        nn.init.zeros_(self.lin.weight)                     # start as identity (gamma=1, beta=0)
        nn.init.zeros_(self.lin.bias)

    def forward(self, h, cond):
        gamma, beta = self.lin(cond).chunk(2, dim=-1)
        if h.dim() == 3:                                    # [B,N,d]: same modulation for every task
            gamma, beta = gamma.unsqueeze(1), beta.unsqueeze(1)
        return (1.0 + gamma) * h + beta


class FiLMPointerPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        d = hp.d_model
        self.d = d
        # task encoder
        self.task_mlp = nn.Sequential(nn.Linear(TASK_DIM, d), nn.ReLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=hp.n_heads, dim_feedforward=hp.ff_dim,
                                           dropout=0.0, batch_first=True)
        self.task_encoder = nn.TransformerEncoder(layer, num_layers=hp.n_enc_layers)
        # preference -> embedding -> (gamma, beta) of the four FiLM layers
        self.pref_mlp = nn.Sequential(nn.Linear(2, hp.film_hidden), nn.ReLU(),
                                      nn.Linear(hp.film_hidden, hp.film_hidden), nn.ReLU())
        self.film1 = FiLM(hp.film_hidden, d)                # task tokens   (shared)
        self.film2 = FiLM(hp.film_hidden, d)                # context       (shared)
        self.film3 = FiLM(hp.film_hidden, d)                # decoder query (actor)
        self.film4 = FiLM(hp.film_hidden, d)                # critic layer  (critic)
        # robot/mission state + context
        self.state_enc = nn.Sequential(nn.Linear(ROBOT_DIM + STATE_DIM, d), nn.ReLU(), nn.Linear(d, d))
        self.ctx_mlp = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU())
        # actor
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d, bias=False)
        self.speed_head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, N_SPEEDS))
        # critic
        self.critic_mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU())
        self.v_head = nn.Linear(d, 1)

    # ------------------------------------------------------------------
    def forward(self, task_feats, robot_feats, state, lam, task_mask):
        """
        task_feats [N,5], robot_feats [8]  : same scenario for the whole batch
        state [B,6], lam [B,2]             : per sample
        task_mask [B,N]                    : True = task cannot be selected
        """
        B, N = state.shape[0], task_feats.shape[0]
        e = self.pref_mlp(lam)                                                     # [B,hid]

        H = self.task_encoder(self.task_mlp(task_feats)[None])                     # [1,N,d]
        H1 = self.film1(H.expand(B, N, -1), e)                                     # FiLM 1

        s = self.state_enc(torch.cat([robot_feats.expand(B, -1), state], dim=-1))  # [B,d]
        free = (~task_mask).float().unsqueeze(-1)                                  # [B,N,1]
        pooled = (H1 * free).sum(1) / free.sum(1).clamp(min=1.0)                   # mean over open tasks
        c = self.film2(self.ctx_mlp(torch.cat([pooled, s], dim=-1)), e)            # FiLM 2

        q = self.film3(self.Wq(c), e)                                              # FiLM 3 (actor)
        k = self.Wk(H1)
        logits = hp.logit_clip * torch.tanh((q.unsqueeze(1) * k).sum(-1) / math.sqrt(self.d))
        logits = logits.masked_fill(task_mask, -1e9)

        z = self.film4(self.critic_mlp(c), e)                                      # FiLM 4 (critic)
        value = self.v_head(z).squeeze(-1)
        return logits, q, H1, value

    # ------------------------------------------------------------------
    def act(self, task_feats, robot_feats, state, lam, ok, task=None, level=None):
        """
        Sample a (task, speed) action, or re-evaluate the given one (PPO update).
        ok [B,N,S] : True = (task, speed) allowed.
        """
        task_mask = ~ok.any(-1)
        logits, q, H1, value = self.forward(task_feats, robot_feats, state, lam, task_mask)

        dist_t = Categorical(logits=logits)
        if task is None:
            task = dist_t.sample()

        rows = torch.arange(task.shape[0], device=task.device)
        speed_mask = ~ok[rows, task]                                               # [B,S]
        h_i = H1[rows, task]                                                       # chosen task embedding
        speed_logits = self.speed_head(torch.cat([q, h_i], dim=-1)).masked_fill(speed_mask, -1e9)
        dist_s = Categorical(logits=speed_logits)
        if level is None:
            level = dist_s.sample()

        logp = dist_t.log_prob(task) + dist_s.log_prob(level)
        entropy = dist_t.entropy() + dist_s.entropy()
        return task, level, logp, entropy, value


# ======================================================================
# PPO helpers
# ======================================================================
def compute_gae(rew, val, valid):
    """rew, val, valid: [T,B]. Episodes end when valid turns False (value after the end = 0)."""
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


def ppo_update(policy, optimizer, data, task_f, robot_f):
    """Runs hp.epochs_per_scenario PPO epochs over the flat batch `data` (valid steps only)."""
    M = data["task"].shape[0]
    adv = data["adv"]
    if M > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = []
    for epoch in range(hp.epochs_per_scenario):
        perm = torch.randperm(M, device=adv.device)
        for i in range(0, M, hp.minibatch_size):
            idx = perm[i:i + hp.minibatch_size]
            _, _, logp, ent, value = policy.act(task_f, robot_f, data["state"][idx], data["lam"][idx],
                                                data["ok"][idx], task=data["task"][idx],
                                                level=data["level"][idx])
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
    return np.mean(stats, axis=0)                                                  # pg, vf, entropy, kl


# ======================================================================
# Checkpoints
# ======================================================================
def save_checkpoint(run_dir, done_iters, policy, optimizer):
    total_epochs = done_iters * hp.epochs_per_scenario
    name = f"{hp.run_id}_it{done_iters:05d}_ep{total_epochs:05d}.pt"
    torch.save({
        "iteration": done_iters,              # scenarios finished
        "total_epochs": total_epochs,         # PPO epochs finished
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "hparams": asdict(hp),
    }, os.path.join(run_dir, name))
    return name


def load_checkpoint(run_dir, policy, optimizer):
    if hp.resume_path:
        path = hp.resume_path
    else:
        files = sorted(glob.glob(os.path.join(run_dir, f"{hp.run_id}_it*.pt")))
        path = files[-1] if files else None
    if path is None:
        print("No checkpoint found -> starting from scratch.")
        return 0

    ckpt = torch.load(path, map_location=hp.device)
    policy.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    skip = {"resume", "resume_path", "device"}
    changed = {k: (v, getattr(hp, k)) for k, v in ckpt["hparams"].items()
               if k not in skip and hasattr(hp, k) and getattr(hp, k) != v}
    print(f"Resumed from {os.path.basename(path)} (iteration {ckpt['iteration']}, "
          f"{ckpt['total_epochs']} epochs).")
    if changed:
        print("Hyperparameters changed since that checkpoint (saved -> now):", changed)
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

    policy = FiLMPointerPolicy().to(dev)
    optimizer = torch.optim.Adam(policy.parameters(), lr=hp.lr)
    start_iter = load_checkpoint(run_dir, policy, optimizer) if hp.resume else 0

    seeds = [hp.start_seed + i for i in range(hp.num_scenarios)] * hp.num_passes
    log_path = os.path.join(run_dir, "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("iter,seed,category,ret,E_wh,T_s,unserved,E_lamE_high,T_lamE_high,"
                    "E_lamT_high,T_lamT_high,pg_loss,v_loss,entropy,kl,energy_check_err\n")

    print(f"Run '{hp.run_id}' on {dev}: {len(seeds)} scenarios, start iteration {start_iter}")

    for it in range(start_iter, len(seeds)):
        seed = seeds[it]
        B = hp.rollouts_per_scenario

        # ---- 1. scenario (one robot, N tasks) ----------------------------------
        robots, task_pos, task_payload, task_service, _, err = generate_scenario(
            hp.n_robots, hp.n_tasks, seed=seed, return_meta=True)
        if err is not None:
            print(f"[it {it}] seed {seed}: scenario error ({err}) -> skipped")
            continue

        # ---- 2. environment, energy model, reward scales, network inputs --------
        env = SingleDroneEnv(robots, task_pos, task_payload, task_service, B)
        env.reset()
        if not env.ok.any():
            print(f"[it {it}] seed {seed}: no task fits the battery -> skipped")
            continue
        E_ref, T_ref = reference_scales(env)
        task_f = task_features(task_pos, task_payload, task_service, env.cap)       # [N,5]
        robot_f = robot_features(robots)                                           # [8]

        # ---- 3. preferences: one lambda per parallel episode --------------------
        lam = Dirichlet(torch.full((2,), hp.lambda_alpha, device=dev)).sample((B,))  # [B,2]

        # ---- 4. rollout --------------------------------------------------------
        keys = ["state", "ok", "task", "level", "logp", "value", "reward", "valid"]
        buf = {k: [] for k in keys}
        for t in range(env.N):
            if not env.active.any():
                break
            state = env.observe()                                                  # dynamic state
            ok = env.ok                                                            # allowed (task, speed)
            valid = env.active.clone()

            with torch.no_grad():                                                  # (a) network -> action
                task, level, logp, _, value = policy.act(task_f, robot_f, state, lam, ok)

            E, T, unserved, _ = env.step(task, level)                              # (b) energy + time

            reward = -(lam[:, 0] * E / E_ref + lam[:, 1] * T / T_ref) \
                     - hp.unserved_penalty * unserved                              # (c) scalarised reward

            for k, v in zip(keys, [state, ok, task, level, logp, value, reward, valid]):
                buf[k].append(v)
        buf = {k: torch.stack(v, 0) for k, v in buf.items()}                       # each [T,B,...]
        Tn = buf["task"].shape[0]

        # ---- 5. feedback: advantages and returns --------------------------------
        adv, ret = compute_gae(buf["reward"], buf["value"], buf["valid"])
        buf["adv"], buf["ret"] = adv, ret
        buf["lam"] = lam.unsqueeze(0).expand(Tn, -1, -1)

        flat_valid = buf["valid"].reshape(-1)
        data = {k: buf[k].reshape(-1, *buf[k].shape[2:])[flat_valid]
                for k in ["state", "lam", "ok", "task", "level", "logp", "adv", "ret"]}

        # ---- 6. PPO update (epochs_per_scenario epochs) -------------------------
        pg, vf, ent, kl = ppo_update(policy, optimizer, data, task_f, robot_f)

        # ---- 7. logging, energy cross-check, checkpoint -------------------------
        err_e = check_energy(env, buf["task"], buf["level"]) if hp.check_energy else float("nan")
        ep_ret = (buf["reward"] * buf["valid"]).sum(0)
        unserved_final = (~env.visited).sum(1).float()
        hi_E, hi_T = lam[:, 0] > 0.5, lam[:, 0] <= 0.5                             # energy- / time-focused episodes
        mean = lambda x, m: x[m].mean().item() if m.any() else float("nan")

        row = [it, seed, int(robots["category"][0]), ep_ret.mean().item(),
               env.E_used.mean().item(), env.T_used.mean().item(), unserved_final.mean().item(),
               mean(env.E_used, hi_E), mean(env.T_used, hi_E),
               mean(env.E_used, hi_T), mean(env.T_used, hi_T), pg, vf, ent, kl, err_e]
        with open(log_path, "a") as f:
            f.write(",".join(f"{x:.5g}" if isinstance(x, float) else str(x) for x in row) + "\n")

        if (it + 1) % 50 == 0 or it + 1 == len(seeds):
            print(f"[it {it + 1}/{len(seeds)} seed {seed} cat {row[2]}] "
                f"return {row[3]:.3f} | E {row[4]:.1f} Wh  T {row[5]:.0f} s  unserved {row[6]:.1f} | "
                f"lamE>0.5: E {row[7]:.1f} T {row[8]:.0f} | lamE<=0.5: E {row[9]:.1f} T {row[10]:.0f} | "
                f"pg {pg:.3f} vf {vf:.3f} ent {ent:.2f} kl {kl:.4f} | energy check {err_e:.1e}")

        if (it + 1) % hp.save_every == 0 or it + 1 == len(seeds):
            name = save_checkpoint(run_dir, it + 1, policy, optimizer)
            print(f"   saved {name}")


if __name__ == "__main__":
    train()