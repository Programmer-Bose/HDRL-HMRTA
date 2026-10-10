"""
drone_energy.py
---------------
Parametric energy model for heterogeneous delivery drones (torch, batched).

Per-robot parameters come from the scenario's robot dict (scenario_gen.py):
    category, payload_kg, flight_min, battery_wh, speed_ms

Model (all per robot, mass changes as payload is delivered):
    mass         m = m_empty + current_load
    hover power  P_hover(m) = P_ref * (m / m_ref)^1.5
                 P_ref = battery_wh / (flight_min/60)   -> rated endurance at full payload
                 m_ref = m_empty + payload_max
    cruise power P(v, m) = P_hover(m) + k_drag * v^3
                 k_drag chosen so drag adds DRAG_FRAC * P_ref at the robot's top speed
    climb        E = m g dh / (eta * 3600)   (descent = 0)
    service      E = P_hover(m) * t_service / 3600  (mass before dropping the payload)

Usage in another script:
    from drone_energy import DroneEnergyModel
    model = DroneEnergyModel(robots)                  # robots dict from generate_scenario
    out = model.tour_energy(robot_ids, coords, payload, service)
    # or, inside a step-by-step environment:
    E, t = model.leg_cost(robot_id, p_from, p_to, load, service_s)
"""

import numpy as np
import torch

# ---- Editable constants ---------------------------------------------------
WORLD_SIZE_M = 1000.0                       # same as scenario_gen.WORLD_SIZE_M
EMPTY_MASS_KG = [6.0, 10.0, 14.0, 18.0]     # airframe + battery mass per category (ASSUMED)
G = 9.81
MOTOR_EFF = 0.85          # only used for the climb term
HOVER_EXP = 1.5           # P_hover ~ mass^1.5 (momentum theory)
DRAG_FRAC = 1.5           # extra power at top speed = DRAG_FRAC * P_ref
SPEED_LEVELS = [0.3, 0.38, 0.46, 0.54, 0.62, 0.7, 0.78, 0.86, 0.93, 1.0]   # action choices: fraction of the robot's max speed
RESERVE_SOC = 0.20        # keep 20 % battery as safety reserve


class DroneEnergyModel:
    def __init__(self, robots, world_size=WORLD_SIZE_M, device="cpu"):
        t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
        cat = torch.as_tensor(np.asarray(robots["category"]), dtype=torch.long, device=device)

        self.world = world_size
        self.payload_max = t(robots["payload_kg"])
        self.battery = t(robots["battery_wh"])
        self.speed = t(robots["speed_ms"])
        self.m_empty = t(EMPTY_MASS_KG)[cat]

        self.m_ref = self.m_empty + self.payload_max
        self.p_ref = self.battery / (t(robots["flight_min"]) / 60.0)    # [W]
        self.k_drag = DRAG_FRAC * self.p_ref / self.speed ** 3

    # ------------------------------------------------------------------
    def hover_power(self, ids, load):
        """Hover power [W] of robot(s) `ids` carrying `load` kg."""
        mass = self.m_empty[ids] + load
        return self.p_ref[ids] * (mass / self.m_ref[ids]) ** HOVER_EXP

    # ------------------------------------------------------------------
    def speed_from_level(self, ids, level):
        """Speed [m/s] for speed-level index `level` (0..len(SPEED_LEVELS)-1)."""
        frac = torch.as_tensor(SPEED_LEVELS, device=self.speed.device)[level]
        return frac * self.speed[ids]

    # ------------------------------------------------------------------
    def leg_cost(self, ids, p_from, p_to, load, service_s, speed=None):
        """
        Energy and time of ONE leg (fly p_from -> p_to, then serve at p_to).
        Works for any batch shape as long as the tensors broadcast.

        ids       : long tensor of robot ids
        p_from/to : [..., 3] normalised coordinates
        load      : payload carried during the leg and service [kg]
        service_s : service time at p_to [s] (0 for return to depot)
        speed     : optional speed [m/s]; default = robot's max speed

        Returns E_wh, time_s  (same shape as `load`)
        """
        seg = p_to - p_from
        dist = torch.norm(seg, dim=-1) * self.world
        climb = torch.clamp(seg[..., 2], min=0.0) * self.world
        v = self.speed[ids] if speed is None else speed

        mass = self.m_empty[ids] + load
        p_hover = self.hover_power(ids, load)
        t_fly = dist / v

        e_travel = (p_hover + self.k_drag[ids] * v ** 3) * t_fly / 3600.0
        e_climb = mass * G * climb / (MOTOR_EFF * 3600.0)
        e_service = p_hover * service_s / 3600.0

        return e_travel + e_climb + e_service, t_fly + service_s

    # ------------------------------------------------------------------
    def tour_energy(self, robot_ids, coords, task_payload, task_service,
                    speeds=None, speed_levels=None):
        """
        Full tours, batched.

        robot_ids    : [B] long
        coords       : [B, N+2, 3] depot, N tasks in visit order, depot
        task_payload : [B, N] kg delivered at each task
        task_service : [B, N] s
        speeds       : [B, N+1] m/s speed on each leg (optional; default = robot max speed)
        speed_levels : [B, N+1] long, index into SPEED_LEVELS (alternative to `speeds`)

        Returns dict: energy_wh [B], soc_drop [B] (%), time_s [B] (makespan),
                      battery_ok [B] bool, payload_ok [B] bool
        """
        ids = robot_ids[:, None]                                     # [B,1]
        if speed_levels is not None:
            speeds = self.speed_from_level(ids, speed_levels)
        B = task_payload.shape[0]
        zeros = torch.zeros(B, 1, device=task_payload.device)

        # load carried on each of the N+1 legs (drops as tasks are delivered)
        total = task_payload.sum(1, keepdim=True)
        delivered = torch.cat([zeros, task_payload.cumsum(1)], dim=1)   # [B,N+1]
        load = total - delivered
        service = torch.cat([task_service, zeros], dim=1)               # no service on return

        E, T = self.leg_cost(ids, coords[:, :-1], coords[:, 1:], load, service, speed=speeds)
        energy = E.sum(1)
        soc_drop = energy / self.battery[robot_ids] * 100.0

        return {
            "energy_wh": energy,
            "soc_drop": soc_drop,
            "time_s": T.sum(1),
            "battery_ok": soc_drop <= (1.0 - RESERVE_SOC) * 100.0,
            "payload_ok": total.squeeze(1) <= self.payload_max[robot_ids],
        }


# ----------------------------------------------------------------------
# Example
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from archive.scenario_gen2 import generate_scenario

    robots, task_pos, task_payload, task_service, _ = generate_scenario(50, 100, seed=43)
    model = DroneEnergyModel(robots)

    # sanity check: hover endurance at full payload should match flight_min
    for c in range(4):
        idx = np.where(robots["category"] == c)[0]
        if len(idx) == 0:
            continue                      # category not present in this scenario
        r = int(idx[0])
        p = model.hover_power(r, model.payload_max[r]).item()
        endurance = model.battery[r].item() / p * 60
        print(f"cat {c}: P_hover(full)={p:.0f} W, endurance={endurance:.1f} min "
              f"(rated {robots['flight_min'][r]:.0f})")

    # one tour per robot: first 4 tasks in index order (just a demo order)
    B, N = 4, 4
    robot_ids = torch.tensor([0, 1, 2, 3])
    tp = torch.as_tensor(task_pos, dtype=torch.float32)
    pay = torch.as_tensor(task_payload, dtype=torch.float32)[:N].repeat(B, 1)
    srv = torch.as_tensor(task_service, dtype=torch.float32)[:N].repeat(B, 1)

    depot = torch.as_tensor(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]), dtype=torch.float32
    )[torch.as_tensor(robots["depot_id"])[robot_ids]][:, None, :]          # [B,1,3]
    coords = torch.cat([depot, tp[:N].repeat(B, 1, 1), depot], dim=1)       # [B,N+2,3]

    out = model.tour_energy(robot_ids, coords, pay, srv)
    for b in range(B):
        print(f"R{b} (cat {robots['category'][b]}): "
              f"E={out['energy_wh'][b]:.1f} Wh, SOC drop={out['soc_drop'][b]:.1f} %, "
              f"T={out['time_s'][b]:.0f} s, battery_ok={out['battery_ok'][b].item()}, "
              f"payload_ok={out['payload_ok'][b].item()}")

    # speed-level demo: same tour, same robot, uniform slow / medium / fast
    print("\nSpeed levels on R3's tour:")
    for lv, frac in enumerate(SPEED_LEVELS):
        lev = torch.full((B, N + 1), lv, dtype=torch.long)
        o = model.tour_energy(robot_ids, coords, pay, srv, speed_levels=lev)
        print(f"  {frac:.2f} x vmax: E={o['energy_wh'][3]:.1f} Wh, T={o['time_s'][3]:.0f} s")