"""Print the key numbers of the committed section Vf runs as Markdown tables.

Per configuration (seeds 0-2 pooled): success rate, final object-goal error and wall-clock time per
MPC step. A run succeeds when it reaches every goal (all four square corners, or the single
planner-comparison goal), with the reach test run_experiment.py applies online. The final error is
the distance from the object's last (x, y) position to the last active goal. The step time sums
diagnostics_iter_time over iterations and averages over steps 1.., skipping step 0's JIT warm-up,
as plot_smpc_summary.py does.
"""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEEDS = (0, 1, 2)
CMP_RUNS = [f"cmp_mppi_k{k}" for k in (200, 400, 800, 1600)]
CMP_RUNS += [f"cmp_icem_k200_it{it}" for it in (1, 2, 3, 4)]
CMP_RUNS += [f"cmp_icem_it3_k{k}" for k in (50, 100, 200, 400)]


def point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Same test as run_experiment.point_to_segment_distance, on (x, y) arrays."""
    d = b - a
    t = np.clip(np.dot(p - a, d) / np.dot(d, d), 0.0, 1.0) if np.dot(d, d) > 1e-12 else 0.0
    return float(np.linalg.norm(p - (a + t * d)))


def summarize(path: Path) -> tuple[int, float, float, int]:
    """Return (goals reached, final error [m], mean step time [s], MPC steps) for one run."""
    with np.load(path, allow_pickle=True) as data:
        xy = data["object_pose"][:, :2]
        goals = data["goal_xy"]
        tol = data["reach_tolerance"]
        step_time = data["diagnostics_iter_time"].astype(float).sum(axis=1)
    reached = sum(point_to_segment_distance(goals[k], xy[k], xy[k + 1]) < tol[k] for k in range(len(goals)))
    return reached, float(np.linalg.norm(xy[-1] - goals[-1])), float(step_time[1:].mean()), len(goals)


def row(name: str, paths: list[Path], num_goals: int) -> str:
    runs = [summarize(p) for p in paths]
    success = sum(r[0] >= num_goals for r in runs)
    err = np.array([r[1] for r in runs]) * 1e3
    t = np.array([r[2] for r in runs])
    steps = np.array([r[3] for r in runs])
    return (
        f"| {name} | {success}/{len(runs)} | {err.mean():.1f} ± {err.std():.1f} | "
        f"{t.mean():.2f} ± {t.std():.2f} | {steps.mean():.0f} |"
    )


HEADER = "| Configuration | Success | Final error [mm] | Time per MPC step [s] | MPC steps |\n| --- | --- | --- | --- | --- |"

if __name__ == "__main__":
    print("Square push:\n\n" + HEADER)
    print(row("iCEM, box", [DATA_DIR / f"square_icem_seed{s}.npz" for s in SEEDS], 4))
    print(row("MPPI K = 800, box", [DATA_DIR / f"square_mppi_k800_white_seed{s}.npz" for s in SEEDS], 4))
    print(row("iCEM, 100 g box", [DATA_DIR / "square_icem_box100g.npz"], 4))
    print(row("iCEM, cylinder", [DATA_DIR / "square_icem_cylinder.npz"], 4))
    print("\nPlanner comparison (20 steps, single goal):\n\n" + HEADER)
    for name in CMP_RUNS:
        print(row(name, [DATA_DIR / "cmp_seeds" / f"seed{s}" / f"{name}.npz" for s in SEEDS], 1))
