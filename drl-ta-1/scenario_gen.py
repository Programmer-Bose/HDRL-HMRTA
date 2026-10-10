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
# Number of tasks per robot category (min, max inclusive), used when n_tasks=None.
# Ranges overlap on purpose, so the task count alone does not reveal the category.
TASKS_PER_CATEGORY = [(4, 6), (6, 12), (12, 16), (16, 20)]
SERVICE_BASE_S = 5.0
SERVICE_PER_KG_S = 3.0
WORLD_SIZE_M = 1000.0

DEPOTS = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0]],
    dtype=float,
)

# ----------------------------------------------------------------------
# Top-layer fleet inventory (per-depot pools, used by the new launch-cost /
# peak-load / depot-reserve objectives). Each depot holds a random number
# of robots per category, drawn from this (min, max inclusive) range. The
# pool is kept larger than the expected number of launches so "reserve
# readiness" is a meaningful objective (if every robot is always needed,
# nothing is ever left in reserve).
# ----------------------------------------------------------------------
ROBOTS_PER_CATEGORY_PER_DEPOT = (3, 7)
# Launch cost per category index 0..3 (low -> high). Strictly increasing,
# matching the category ordering (bigger drones cost more to launch).
LAUNCH_COST = [1.0, 2.0, 4.0, 8.0]


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


def generate_scenario(n_robots, n_tasks=None, seed=None, *, return_meta=False,
                      robot_category=None):
    """Create a robot-task scenario.

    n_tasks=None -> the number of tasks is drawn from TASKS_PER_CATEGORY of each
    robot's category (for one robot: a single range). Pass an int to fix it.
    robot_category=None -> sample categories randomly as before. Otherwise, force
    all robots in the scenario to the requested category index (0..3).

    Returns the original five outputs by default. If return_meta=True,
    an additional error message is returned as the last element.
    """
    try:
        n_robots = int(n_robots)
        n_tasks = None if n_tasks is None else int(n_tasks)
        robot_category = None if robot_category is None else int(robot_category)
    except (TypeError, ValueError):
        error_msg = "n_robots, n_tasks, and robot_category must be integers."
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

    n_categories = len(CATEGORIES["payload_kg"])
    if (n_robots <= 0 or (n_tasks is not None and n_tasks <= 0)
            or (robot_category is not None and not 0 <= robot_category < n_categories)):
        error_msg = (
            "n_robots and n_tasks must be greater than 0, and robot_category "
            f"must be between 0 and {n_categories - 1}."
        )
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

    depot_id = np.arange(n_robots) % len(DEPOTS)

    if robot_category is None:
        probs = rng.dirichlet(np.ones(n_categories) * 5)
        base_size = int((depot_id == 0).sum())                 # robots in depot 0 (reference count)
        template = rng.choice(n_categories, size=base_size, p=probs)   # same multiset for every depot
        categories = np.empty(n_robots, dtype=int)
        for d in range(len(DEPOTS)):
            idx = np.where(depot_id == d)[0]
            cats = np.resize(template, len(idx))               # trim/pad if a depot's size differs
            categories[idx] = rng.permutation(cats)
    else:
        probs = np.eye(n_categories)[robot_category]
        categories = np.full(n_robots, robot_category, dtype=int)

    robots = {
        "category": categories,
        "payload_kg": np.array(CATEGORIES["payload_kg"])[categories],
        "flight_min": np.array(CATEGORIES["flight_min"])[categories],
        "battery_wh": np.array(CATEGORIES["battery_wh"])[categories],
        "speed_ms": np.array(CATEGORIES["speed_ms"])[categories],
        "depot_id": depot_id,
    }

    if n_tasks is None:      # task count follows the robot category
        n_tasks = int(sum(
            rng.integers(TASKS_PER_CATEGORY[c][0], TASKS_PER_CATEGORY[c][1] + 1)
            for c in categories
        ))

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


def generate_fleet_inventory(seed=None, n_depots=None,
                              count_range=ROBOTS_PER_CATEGORY_PER_DEPOT):
    """Build the top-layer robot POOL: every depot gets a random number of
    robots of EVERY category (counts can differ across depots), independent
    of any task scenario. This is the fleet the assignment policy chooses
    from -- only some of these robots end up launched (y_r = 1).

    Returns a `robots` dict with the same fields/shape convention as
    `generate_scenario` (category, payload_kg, flight_min, battery_wh,
    speed_ms, depot_id), plus `launch_cost` (per-robot, from LAUNCH_COST).
    Robots are ordered depot-by-depot, category-by-category.
    """
    rng = np.random.default_rng(seed)
    n_categories = len(CATEGORIES["payload_kg"])
    if n_depots is None:
        n_depots = len(DEPOTS)

    category_list, depot_list = [], []
    counts = rng.integers(count_range[0], count_range[1] + 1, size=n_categories)   # same for every depot
    for d in range(n_depots):
        for c, n in enumerate(counts):
            category_list += [c] * int(n)
            depot_list += [d] * int(n)

    categories = np.array(category_list, dtype=int)
    depot_id = np.array(depot_list, dtype=int)

    robots = {
        "category": categories,
        "payload_kg": np.array(CATEGORIES["payload_kg"])[categories],
        "flight_min": np.array(CATEGORIES["flight_min"])[categories],
        "battery_wh": np.array(CATEGORIES["battery_wh"])[categories],
        "speed_ms": np.array(CATEGORIES["speed_ms"])[categories],
        "depot_id": depot_id,
        "launch_cost": np.array(LAUNCH_COST)[categories],
    }
    return robots


def generate_fleet_tasks(robots, n_tasks, seed=None, payload_usage_range=(0.50, 0.70)):
    """Task set sized against the WHOLE fleet pool's capacity (not just the
    robots that end up launched -- we don't know that yet). Lower
    payload_usage than `generate_scenario` by default, since the pool is
    deliberately oversized versus `generate_scenario`'s single-fleet case.

    Returns (task_pos, task_payload, task_service, error_message).
    """
    rng = np.random.default_rng(seed)
    task_pos, task_error = sample_task_points(rng, n_tasks, return_meta=True)
    if task_error is not None:
        warnings.warn(task_error, RuntimeWarning)
        return task_pos, np.zeros(0, dtype=float), np.zeros(0, dtype=float), task_error

    payload_usage = rng.uniform(*payload_usage_range)
    total_payload = payload_usage * robots["payload_kg"].sum()
    shares = rng.dirichlet(np.ones(n_tasks) * 5)
    task_payload = total_payload * shares
    task_service = SERVICE_BASE_S + SERVICE_PER_KG_S * task_payload
    return task_pos, task_payload, task_service, None


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
    n_robots = 25
    n_tasks = 100

    robots, task_pos, task_payload, task_service, probs, error = generate_scenario(
        n_robots,
        n_tasks,
        seed=42,
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