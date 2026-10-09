"""
test_policy.py
--------------
Loads a trained checkpoint and checks whether the policy really reacts to the
preference lambda = [lambda_E, lambda_T].

For every test scenario (unseen seeds, one drone):
    - the SAME scenario is solved for N_LAMBDAS values of lambda_E from 0 (only time) to 1 (only energy)
    - energy, makespan, visiting order and speeds are recorded
Outputs:
    - a text summary with a PASS / FAIL verdict
    - Pareto plot (energy vs makespan) per scenario
    - 3D plots of the visiting sequence for chosen lambda values
    - test_summary.csv

Files needed in the same folder: train_ppo.py, scenario_gen.py, drone_energy.py
"""

import os
import glob
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import train_ppo as tp
from scenario_gen import generate_scenario
from drone_energy import SPEED_LEVELS

# ======================================================================
# TEST SETTINGS  (edit here only)
# ======================================================================
RUN_ID = "run_004"
SAVE_ROOT = "checkpoints"
CKPT_PATH = ""                    # "" = latest checkpoint of RUN_ID, or give a .pt path
TEST_START_SEED = 30000          # unseen seeds (training used 43 ... 142)
NUM_TEST_SCENARIOS = 10
TEST_N_TASKS = None               # None = task count follows the robot category
N_LAMBDAS = 11                    # lambda_E = 0, 0.1, ..., 1
GREEDY = True                     # True = argmax actions, False = sample
PLOT_LAMBDAS_E = [0.0, 0.5, 1.0]  # lambda_E values drawn in the 3D plots
SAVE_FIGS = False
SHOW_FIGS = False
OUT_DIR = os.path.join("test_results", RUN_ID)
PASS_THRESHOLD_PCT = 1.0          # energy-focused must save >= this % energy and lose >= this % time


# ======================================================================
# Model loading
# ======================================================================
def load_model():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if CKPT_PATH:
        path = CKPT_PATH
    else:
        files = sorted(glob.glob(os.path.join(SAVE_ROOT, RUN_ID, f"{RUN_ID}_it*.pt")))
        assert files, f"No checkpoint found in {os.path.join(SAVE_ROOT, RUN_ID)}"
        path = files[-1]

    ckpt = torch.load(path, map_location=dev)
    for k, v in ckpt["hparams"].items():              # use the network sizes etc. used in training
        if hasattr(tp.hp, k):
            setattr(tp.hp, k, v)
    tp.hp.device = dev

    policy = tp.FiLMPointerPolicy().to(dev)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    print(f"Loaded {os.path.basename(path)} (iteration {ckpt['iteration']}, "
          f"{ckpt['total_epochs']} epochs) on {dev}")
    return policy


# ======================================================================
# Rollout: one episode per lambda, all on the same scenario
# ======================================================================
@torch.no_grad()
def rollout(policy, env, task_f, robot_f, lam):
    env.reset()
    tasks, levels, valid = [], [], []
    for t in range(env.N):
        if not env.active.any():
            break
        state, ok = env.observe(), env.ok
        valid.append(env.active.clone())

        if GREEDY:
            logits, q, H1, _ = policy(task_f, robot_f, state, lam, ~ok.any(-1))
            task = logits.argmax(-1)
            rows = torch.arange(env.B, device=env.dev)
            speed_logits = policy.speed_head(torch.cat([q, H1[rows, task]], dim=-1))
            level = speed_logits.masked_fill(~ok[rows, task], -1e9).argmax(-1)
        else:
            task, level, _, _, _ = policy.act(task_f, robot_f, state, lam, ok)

        env.step(task, level)
        tasks.append(task)
        levels.append(level)
    return (torch.stack(tasks).cpu().numpy(), torch.stack(levels).cpu().numpy(),
            torch.stack(valid).cpu().numpy())                                     # each [T,B]


# ======================================================================
# Summary / verdict
# ======================================================================
def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 0 else float("nan")


def summarise(results):
    print("\n=== Per scenario: lambda_E = 0 (time) vs lambda_E = 1 (energy) ===")
    print(f"{'seed':>6} {'cat':>3} {'N':>3} | {'E(0)':>7} {'E(1)':>7} {'dE%':>7} | "
          f"{'T(1)':>6} {'T(0)':>6} {'dT%':>7} | {'v(0)':>5} {'v(1)':>5} | "
          f"{'unserved':>8} | order differs")
    dE, dT, cE, cT = [], [], [], []
    for r in results:
        de = (r["E"][-1] - r["E"][0]) / r["E"][0] * 100
        dt = (r["T"][-1] - r["T"][0]) / r["T"][0] * 100
        v0 = np.mean([SPEED_LEVELS[l] for l in r["levels"][0]])
        v1 = np.mean([SPEED_LEVELS[l] for l in r["levels"][-1]])
        differs = r["orders"][0] != r["orders"][-1]
        dE.append(de)
        dT.append(dt)
        cE.append(corr(r["lamE"], r["E"]))
        cT.append(corr(r["lamE"], r["T"]))
        print(f"{r['seed']:>6} {r['cat']:>3} {r['n']:>3} | {r['E'][0]:7.1f} {r['E'][-1]:7.1f} {de:+7.1f} | "
              f"{r['T'][0]:6.0f} {r['T'][-1]:6.0f} {dt:+7.1f} | {v0:5.2f} {v1:5.2f} | "
              f"{int(r['unserved'][0]):>3}/{int(r['unserved'][-1]):<4} | {differs}")

    mdE, mdT = np.nanmean(dE), np.nanmean(dT)
    mcE, mcT = np.nanmean(cE), np.nanmean(cT)
    print("\n=== Verdict ===")
    print(f"Mean change when going from lambda_E=0 to 1: energy {mdE:+.1f} %, makespan {mdT:+.1f} %")
    print(f"Mean correlation with lambda_E: energy {mcE:+.2f} (want < 0), makespan {mcT:+.2f} (want > 0)")
    passed = (mdE < -PASS_THRESHOLD_PCT) and (mdT > PASS_THRESHOLD_PCT) and (mcE < 0) and (mcT > 0)
    if passed:
        print("PASS: the policy trades energy against time according to the preference.")
    else:
        print("FAIL: no clear preference dependence yet -> train longer / check the speed trade-off.")
    print("Note: if 'unserved' differs between the two columns, the comparison is not like for like.")


def save_csv(results):
    with open(os.path.join(OUT_DIR, "test_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "category", "n_tasks", "lambda_E", "energy_wh", "time_s",
                    "unserved", "order", "speed_levels"])
        for r in results:
            for i, le in enumerate(r["lamE"]):
                w.writerow([r["seed"], r["cat"], r["n"], f"{le:.2f}", f"{r['E'][i]:.2f}",
                            f"{r['T'][i]:.1f}", int(r["unserved"][i]),
                            " ".join(map(str, r["orders"][i])), " ".join(map(str, r["levels"][i]))])


# ======================================================================
# Plots
# ======================================================================
def plot_pareto(results):
    n = len(results)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    sc = None
    for ax, r in zip(axes.ravel(), results):
        ax.plot(r["T"], r["E"], "-", color="gray", lw=0.8, zorder=1)
        sc = ax.scatter(r["T"], r["E"], c=r["lamE"], cmap="coolwarm_r", s=45, zorder=2)
        ax.scatter([r["T_nn"]], [r["E_nn"]], marker="*", s=150, c="black", zorder=3,
                   label="nearest-neighbour, max speed")
        ax.set_xlabel("makespan T [s]")
        ax.set_ylabel("energy E [Wh]")
        ax.set_title(f"seed {r['seed']}, cat {r['cat']}, {r['n']} tasks")
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=8)
    fig.colorbar(sc, ax=axes.ravel().tolist(), label="lambda_E  (red = energy, blue = time)")
    fig.suptitle("Energy vs makespan for lambda_E = 0 ... 1 (same scenario)")
    if SAVE_FIGS:
        fig.savefig(os.path.join(OUT_DIR, "pareto.png"), dpi=150, bbox_inches="tight")


def draw_tour(ax, pos, depot, order, levels, ret_level, title):
    """3D path: depot -> tasks in visiting order -> depot. Leg colour = speed level."""
    cmap = plt.cm.viridis
    S = len(SPEED_LEVELS)
    pts = [depot] + [pos[j] for j in order] + [depot]
    leg_levels = list(levels) + [ret_level]
    for k in range(len(pts) - 1):
        xs, ys, zs = zip(pts[k], pts[k + 1])
        ax.plot(xs, ys, zs, color=cmap(leg_levels[k] / max(S - 1, 1)), lw=2.5)

    ax.scatter(*pos.T, c="black", s=25)
    ax.scatter(*depot, marker="s", s=110, c="purple")
    ax.text(*depot, " depot", fontsize=8, color="purple")
    for rank, j in enumerate(order):
        ax.text(pos[j][0], pos[j][1], pos[j][2], f" {rank + 1}", fontsize=10)
    left_out = [j for j in range(len(pos)) if j not in order]
    if left_out:
        ax.scatter(*pos[left_out].T, marker="x", c="red", s=60)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, fontsize=10)


def plot_tours(r, show_idx):
    n = len(show_idx)
    fig = plt.figure(figsize=(6 * n, 6))
    for i, b in enumerate(show_idx):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        draw_tour(ax, r["pos"], r["depot"], r["orders"][b], r["levels"][b], r["ret_level"],
                  f"lambda_E={r['lamE'][b]:.1f}   E={r['E'][b]:.0f} Wh   T={r['T'][b]:.0f} s")
    S = len(SPEED_LEVELS)
    handles = [Line2D([0], [0], color=plt.cm.viridis(i / max(S - 1, 1)), lw=3,
                      label=f"{f:.2f} x vmax") for i, f in enumerate(SPEED_LEVELS)]
    handles += [Line2D([0], [0], marker="x", color="red", lw=0, label="not served")]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), fontsize=9)
    fig.suptitle(f"Visiting sequence (numbers) - seed {r['seed']}, category {r['cat']}")
    if SAVE_FIGS:
        fig.savefig(os.path.join(OUT_DIR, f"tours_seed{r['seed']}.png"), dpi=150, bbox_inches="tight")


# ======================================================================
# Main
# ======================================================================
def main():
    policy = load_model()
    dev = tp.hp.device
    os.makedirs(OUT_DIR, exist_ok=True)

    lamE = torch.linspace(0, 1, N_LAMBDAS, device=dev)
    lam = torch.stack([lamE, 1 - lamE], dim=1)                                   # [B,2]
    lamE_np = lamE.cpu().numpy()
    show_idx = [int(np.argmin(np.abs(lamE_np - x))) for x in PLOT_LAMBDAS_E]

    results = []
    for s in range(NUM_TEST_SCENARIOS):
        seed = TEST_START_SEED + s

        # ---- 1. scenario + environment ---------------------------------------
        robots, pos, pay, srv, _, err = generate_scenario(1, TEST_N_TASKS, seed=seed, return_meta=True)
        if err is not None:
            print(f"seed {seed}: scenario error -> skipped")
            continue
        env = tp.SingleDroneEnv(robots, pos, pay, srv, N_LAMBDAS)
        env.reset()
        if not env.ok.any():
            print(f"seed {seed}: no task fits the battery -> skipped")
            continue
        task_f = tp.task_features(pos, pay, srv, env.cap)
        robot_f = tp.robot_features(robots)

        # ---- 2. same scenario, every lambda, through the network ---------------
        tasks, levels, valid = rollout(policy, env, task_f, robot_f, lam)

        # ---- 3. collect results -------------------------------------------------
        E_nn, T_nn = tp.reference_scales(env)
        T_steps = tasks.shape[0]
        orders = [[int(tasks[t, b]) for t in range(T_steps) if valid[t, b]] for b in range(N_LAMBDAS)]
        lvls = [[int(levels[t, b]) for t in range(T_steps) if valid[t, b]] for b in range(N_LAMBDAS)]
        results.append(dict(
            seed=seed, cat=int(robots["category"][0]), n=env.N, lamE=lamE_np,
            E=env.E_used.cpu().numpy(), T=env.T_used.cpu().numpy(),
            unserved=(~env.visited).sum(1).cpu().numpy(),
            orders=orders, levels=lvls, pos=pos, depot=env.depot.cpu().numpy(),
            ret_level=tp.hp.return_speed_level, E_nn=E_nn, T_nn=T_nn))

    if not results:
        print("No usable test scenario.")
        return

    # ---- 4. verdict + plots --------------------------------------------------------
    summarise(results)
    save_csv(results)
    plot_pareto(results)
    for r in results:
        plot_tours(r, show_idx)
    if SHOW_FIGS:
        plt.show()


if __name__ == "__main__":
    main()
