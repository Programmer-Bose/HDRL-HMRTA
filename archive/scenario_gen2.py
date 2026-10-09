import warnings

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Robot categories: index 0 (low) -> 3 (high). Edit values freely.
# ----------------------------------------------------------------------
CATEGORIES = {
    "payload_kg": [2.0, 4.0, 8.0, 16.0],
    "flight_min": [20.0, 40.0, 60.0, 80.0],
    "battery_wh": [600.0, 1260.0, 1940.0, 2600.0],
    "speed_ms": [10.0, 13.0, 16.0, 20.0],
}
TASK_Z_RANGE = (0.4, 0.8)
MIN_TASK_DIST = 0.1
SERVICE_BASE_S = 5.0
SERVICE_PER_KG_S = 3.0
WORLD_SIZE_M = 1000.0

DEPOTS = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
    dtype=float,
)


def sample_task_points(rng, n_tasks, max_tries=100000, *, return_meta=False):
    """Return valid task points without raising errors.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator instance.
    n_tasks : int
        Number of task points to create.
    max_tries : int
        Maximum attempts before giving up.
    return_meta : bool
        If True, return (points, error_message) instead of only the array.
    """
    if rng is None:
        rng = np.random.default_rng()

    try:
        n_tasks = int(n_tasks)
    except (TypeError, ValueError):
        error_msg = "n_tasks must be an integer."
        empty_points = np.empty((0, 3), dtype=float)
        return (empty_points, error_msg) if return_meta else empty_points

    if n_tasks <= 0:
        error_msg = "n_tasks must be greater than 0."
        empty_points = np.empty((0, 3), dtype=float)
        return (empty_points, error_msg) if return_meta else empty_points

    points = []
    tries = 0
    while len(points) < n_tasks:
        tries += 1
        if tries > max_tries:
            error_msg = (
                f"Could not place {n_tasks} task points with minimum distance "
                f"{MIN_TASK_DIST}. Use fewer tasks or a smaller MIN_TASK_DIST."
            )
            warnings.warn(error_msg, RuntimeWarning)
            empty_points = np.empty((0, 3), dtype=float)
            return (empty_points, error_msg) if return_meta else empty_points

        p = np.array(
            [
                rng.uniform(0, 1),
                rng.uniform(0, 1),
                rng.uniform(*TASK_Z_RANGE),
            ],
            dtype=float,
        )
        if all(np.linalg.norm(p - q) >= MIN_TASK_DIST for q in points):
            points.append(p)

    points_array = np.asarray(points, dtype=float)
    return (points_array, None) if return_meta else points_array


def generate_scenario(n_robots, n_tasks, seed=None, *, return_meta=False):
    """Create a robot-task scenario.

    Returns the original five outputs by default. If return_meta=True,
    an additional error message is returned as the last element.
    """
    try:
        n_robots = int(n_robots)
        n_tasks = int(n_tasks)
    except (TypeError, ValueError):
        error_msg = "n_robots and n_tasks must be integers."
        warnings.warn(error_msg, RuntimeWarning)
        empty_robots = {
            "category": np.array([], dtype=int),
            "payload_kg": np.array([], dtype=float),
            "flight_min": np.array([], dtype=float),
            "battery_wh": np.array([], dtype=float),
            "speed_ms": np.array([], dtype=float),
            "depot_id": np.array([], dtype=int),
        }
        empty_points = np.empty((0, 3), dtype=float)
        empty_payload = np.array([], dtype=float)
        empty_service = np.array([], dtype=float)
        probs = np.array([], dtype=float)
        if return_meta:
            return (
                empty_robots,
                empty_points,
                empty_payload,
                empty_service,
                probs,
                error_msg,
            )
        return empty_robots, empty_points, empty_payload, empty_service, probs

    if n_robots <= 0 or n_tasks <= 0:
        error_msg = "n_robots and n_tasks must be greater than 0."
        warnings.warn(error_msg, RuntimeWarning)
        empty_robots = {
            "category": np.array([], dtype=int),
            "payload_kg": np.array([], dtype=float),
            "flight_min": np.array([], dtype=float),
            "battery_wh": np.array([], dtype=float),
            "speed_ms": np.array([], dtype=float),
            "depot_id": np.array([], dtype=int),
        }
        empty_points = np.empty((0, 3), dtype=float)
        empty_payload = np.array([], dtype=float)
        empty_service = np.array([], dtype=float)
        probs = np.array([], dtype=float)
        if return_meta:
            return (
                empty_robots,
                empty_points,
                empty_payload,
                empty_service,
                probs,
                error_msg,
            )
        return empty_robots, empty_points, empty_payload, empty_service, probs

    rng = np.random.default_rng(seed)

    probs = rng.dirichlet(np.ones(4) * 5)
    categories = rng.choice(4, size=n_robots, p=probs)

    robots = {
        "category": categories,
        "payload_kg": np.array(CATEGORIES["payload_kg"])[categories],
        "flight_min": np.array(CATEGORIES["flight_min"])[categories],
        "battery_wh": np.array(CATEGORIES["battery_wh"])[categories],
        "speed_ms": np.array(CATEGORIES["speed_ms"])[categories],
        "depot_id": np.arange(n_robots) % 4,
    }

    task_pos, task_error = sample_task_points(rng, n_tasks, return_meta=True)
    if task_error is not None:
        warnings.warn(task_error, RuntimeWarning)
        empty_payload = np.zeros(0, dtype=float)
        empty_service = np.zeros(0, dtype=float)
        if return_meta:
            return robots, task_pos, empty_payload, empty_service, probs, task_error
        return robots, task_pos, empty_payload, empty_service, probs

    payload_usage = rng.uniform(0.80, 0.90)      # varies with the seed
    total_payload = payload_usage * robots["payload_kg"].sum()
    shares = rng.dirichlet(np.ones(n_tasks) * 5)
    task_payload = total_payload * shares
    task_service = SERVICE_BASE_S + SERVICE_PER_KG_S * task_payload

    if return_meta:
        return robots, task_pos, task_payload, task_service, probs, None
    return robots, task_pos, task_payload, task_service, probs


def travel_time_s(robots, robot_id, p_from, p_to):
    """Flight time in seconds between two points of the unit cube."""
    speed = robots.get("speed_ms")
    if speed is None:
        warnings.warn("robots dictionary is missing 'speed_ms'.", RuntimeWarning)
        return np.nan

    try:
        robot_speed = float(speed[robot_id])
    except (TypeError, IndexError, KeyError):
        warnings.warn(
            f"Robot id {robot_id} is invalid for the provided robot set.",
            RuntimeWarning,
        )
        return np.nan

    dist_m = np.linalg.norm(np.asarray(p_to) - np.asarray(p_from)) * WORLD_SIZE_M
    return dist_m / robot_speed


def plot_scenario(robots, task_pos, task_payload, task_service):
    if robots is None or not isinstance(robots, dict):
        warnings.warn("robots must be a dictionary. Nothing to plot.", RuntimeWarning)
        return None

    if len(task_pos) == 0 and len(task_payload) == 0 and len(task_service) == 0:
        warnings.warn("No scenario data available to plot.", RuntimeWarning)
        return None

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]

    ax.scatter(*task_pos.T, c="black", marker="o", s=30, label="Task points")
    for i, (p, w, t) in enumerate(zip(task_pos, task_payload, task_service)):
        ax.text(p[0], p[1], p[2], f" T{i}\n {w:.2f} kg\n {t:.0f} s", fontsize=8)

    ax.scatter(*DEPOTS.T, c="purple", marker="s", s=120, label="Depots")
    for d, pos in enumerate(DEPOTS):
        ids = np.where(robots["depot_id"] == d)[0]
        cats = [f"R{i}(C{robots['category'][i]})" for i in ids]
        ax.text(pos[0], pos[1], pos[2] - 0.05,
                f"Depot {d}\n" + ", ".join(cats), fontsize=8, color="purple")

    for c in range(4):
        idx = np.where(robots["category"] == c)[0]
        if len(idx) == 0:
            continue
        pts = DEPOTS[robots["depot_id"][idx]] + 0.02 * (idx[:, None] % 5)
        ax.scatter(*pts.T, c=colors[c], s=40, marker="^",
                   label=(f"Cat {c}: {CATEGORIES['payload_kg'][c]:.0f} kg, "
                          f"{CATEGORIES['speed_ms'][c]:.0f} m/s, "
                          f"{CATEGORIES['battery_wh'][c]:.0f} Wh"))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_title("Heterogeneous MRTA scenario")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()


def main():
    n_robots = 50
    n_tasks = 100

    robots, task_pos, task_payload, task_service, probs, error = generate_scenario(
        n_robots,
        n_tasks,
        seed=43,
        return_meta=True,
    )

    if error is not None:
        print(f"Scenario warning: {error}")

    print("Category probabilities:", np.round(probs, 2))
    print("\nRobots:")
    for i in range(n_robots):
        if i >= len(robots["category"]):
            break
        print(f"  R{i}: cat={robots['category'][i]}, "
              f"payload={robots['payload_kg'][i]:.1f} kg, "
              f"flight={robots['flight_min'][i]:.0f} min, "
              f"battery={robots['battery_wh'][i]:.0f} Wh, "
              f"speed={robots['speed_ms'][i]:.0f} m/s, "
              f"depot={robots['depot_id'][i]}")
    print(f"\nFleet capacity : {robots['payload_kg'].sum():.2f} kg")
    print(f"Total task load: {task_payload.sum():.2f} kg "
          f"({task_payload.sum() / robots['payload_kg'].sum():.0%})")

    print(f"Total service time: {task_service.sum():.0f} s over {n_tasks} tasks")

    print("\nTasks (payload -> service time):")
    for i in range(n_tasks):
        if i >= len(task_payload):
            break
        print(f"  T{i}: {task_payload[i]:.2f} kg -> {task_service[i]:.1f} s")

    plot_scenario(robots, task_pos, task_payload, task_service)


if __name__ == "__main__":
    main()