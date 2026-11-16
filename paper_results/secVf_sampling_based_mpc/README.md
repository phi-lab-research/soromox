# Section Vf: Sampling-Based MPC

This case runs closed-loop sampling-based MPC, iCEM by default and MPPI with
`--mppi`, in which a tapered four-tendon soft arm on a kinematically driven
floating base pushes a box or cylinder around a 10 cm square. The arm is
defined in `code/four_tendon_soft_arm.py`, and the planner primitives and push
cost live in `code/smpc.py`. Each replan samples rollouts against the full
simulation, which runs on [newton-cosserat](https://github.com/tud-phi/newton-cosserat):
it couples the SoRoMoX Cosserat rod model of the arm to the rigid-body physics
of [Newton](https://github.com/newton-physics/newton), and resolves the
arm-object-table contacts with its SAP contact solver.

## Environment

This case has two uv projects, one for the experiments and one for the
figures. Run every command from this directory:

```bash
cd paper_results/secVf_sampling_based_mpc
uv sync --project env   # experiments
uv sync --project .              # renderers and summary figure
```

Both projects use this checkout's SoRoMoX through `override-dependencies`.

The experiment project, [`env/pyproject.toml`](env/pyproject.toml),
pins the stack the paper data was generated on:

| Package | Version |
| --- | --- |
| `newton-cosserat` | [`phi-lab-research/newton-cosserat`](https://github.com/phi-lab-research/newton-cosserat) `main` (Newton 1.7) |
| `newton` | `1.7.0` |
| `warp-lang` | `1.18.0` |
| `jax` | `0.11.1` |

The figure project, [`pyproject.toml`](pyproject.toml), stays on Newton 1.6
because Newton 1.7's RTX viewer requires `ovrtx` 0.5, which renders the sample
clouds black:

| Package | Version |
| --- | --- |
| `newton-cosserat` | [`tud-phi/newton-cosserat`](https://github.com/tud-phi/newton-cosserat) `main` |
| `newton` | `1.6.0.dev0`, git rev `7c677b95` (adds `log_mesh` opacity for the sample clouds) |
| `ovrtx` | `0.4.1` |
| `warp-lang` | `1.16.0` |

Experiments run on the CPU or a GPU (`--device cpu` or `--device cuda:0`).
Both renderers need an RTX GPU. On a machine without a GPU, set
`JAX_PLATFORMS=cpu`; otherwise the JAX CUDA 13 plugin fails at import.

## Experiments

Run one closed-loop push with `code/run_experiment.py`:

```bash
uv run --project env python code/run_experiment.py --device cuda:0 \
  --stop-after-one-square --object-shape box --k-samples 200 100 50 --num-iters 3 --seed 0 \
  --save-path /tmp/soromox-secVf/square_icem_seed0.npz
```

`--k-samples` takes one value or a per-iteration schedule. `--save-samples`
also stores every candidate rollout, which the sample-cloud renderer needs; the
files are large.

Regenerate the complete paper data set:

```bash
DEVICE=cuda:0 code/generate_paper_results.sh
```

The script runs three parts, each over seeds 0, 1 and 2:

- a 20-step planner comparison toward the single push goal (0.05, -0.10) m,
  written to
  `data/cmp_seeds/seed*/`: an MPPI sample-count sweep (`cmp_mppi_k*.npz`,
  K = 200 to 1600, white noise), an iCEM iteration sweep at K = 200
  (`cmp_icem_k200_it*.npz`, 1 to 4 iterations), and an iCEM sample-count sweep
  at 3 iterations (`cmp_icem_it3_k*.npz`, K = 50 to 400);
- full square pushes with iCEM (`data/square_icem_seed*.npz`) and MPPI with
  K = 800 (`data/square_mppi_k800_white_seed*.npz`); the seed-0 iCEM run is
  saved with `--save-samples`;
- two seed-0 iCEM square-push variants, a 100 g box
  (`data/square_icem_box100g.npz`) and a cylinder
  (`data/square_icem_cylinder.npz`).

The script replaces same-named files in `data/`. Set `RESULTS_DIR` to stage a
run elsewhere. Of the committed runs, only `data/square_icem_seed0.npz` keeps
the per-sample arrays; it is stored in Git LFS (`git lfs pull` fetches it).

## Key numbers

The tables below summarize the committed runs. Recompute them with:

```bash
uv run --project . python code/summarize_runs.py
```

A run succeeds when it reaches every goal (all four square corners, or the
single comparison goal) under the online reach test of `run_experiment.py`:
the object passes within 10 % of the leg length of the goal. The final error is
the distance from the object's last (x, y) position to the last active goal.
The time per MPC step is the wall-clock planning time of one replan, rollouts
included, averaged over steps 1 onward to skip the JIT warm-up of step 0.
Values are mean ± standard deviation over seeds 0, 1 and 2; the 100 g box and
cylinder runs use seed 0 only. Square pushes stop after the last corner or
after 100 MPC steps.

Square push:

| Configuration | Success | Final error [mm] | Time per MPC step [s] | MPC steps |
| --- | --- | --- | --- | --- |
| iCEM, box | 3/3 | 8.0 ± 1.3 | 26.54 ± 0.20 | 51 |
| MPPI K = 800, box | 1/3 | 15.9 ± 9.3 | 8.50 ± 0.84 | 85 |
| iCEM, 100 g box | 0/1 | 10.1 | 29.42 | 100 |
| iCEM, cylinder | 1/1 | 5.6 | 25.58 | 53 |

The failed MPPI seeds 1 and 2 reach only the first corner; the 100 g box
reaches three of four corners within 100 steps.

Planner comparison (20 steps toward the single goal):

| Configuration              | Success | Final error [mm] | Time per MPC step [s] |
| -------------------------- | ------- | ---------------- | --------------------- |
| MPPI K = 200               | 2/3     | 11.8 ± 7.6       | 8.39 ± 0.75           |
| MPPI K = 400               | 1/3     | 13.0 ± 4.9       | 8.72 ± 1.11           |
| MPPI K = 800               | 3/3     | 5.9 ± 0.8        | 12.93 ± 1.48          |
| MPPI K = 1600              | 3/3     | 3.6 ± 0.6        | 29.53 ± 0.70          |
| iCEM K = 200, 1 iteration  | 2/3     | 34.4 ± 46.4      | 8.31 ± 0.49           |
| iCEM K = 200, 2 iterations | 3/3     | 6.5 ± 1.6        | 19.32 ± 0.64          |
| iCEM K = 200, 3 iterations | 3/3     | 8.0 ± 1.5        | 31.62 ± 0.79          |
| iCEM K = 200, 4 iterations | 3/3     | 4.1 ± 0.4        | 44.82 ± 0.21          |
| iCEM 3 iterations, K = 50  | 3/3     | 4.2 ± 1.2        | 19.02 ± 0.55          |
| iCEM 3 iterations, K = 100 | 3/3     | 5.9 ± 1.2        | 23.07 ± 0.70          |
| iCEM 3 iterations, K = 200 | 3/3     | 8.0 ± 1.5        | 30.86 ± 0.58          |
| iCEM 3 iterations, K = 400 | 3/3     | 5.8 ± 2.4        | 41.05 ± 1.54          |

## Paper artifacts

Recreate the summary figure from the committed data and renders:

```bash
uv run --project . python code/plot_smpc_summary.py
```

This writes `../final_outputs/sampling_based_mpc.pdf`. Panel A shows RTX renders of the sample
cloud across iCEM iterations, B the MPPI-versus-iCEM planning-time and
performance trade-off, C RTX renders of the Franka-mounted arm pushing along the
trajectory, and D the top-down square-push trajectories of iCEM and MPPI
K = 800 over three seeds each. Panel B averages the seeds under
`data/cmp_seeds/`. LaTeX is used for text when it is installed.

Render the panel A frames, one per iCEM iteration of MPC step 0, from a run
saved with `--save-samples`:

```bash
code/render_sample_cloud.sh data/square_icem_seed0.npz
```

An optional second argument selects the drawn horizon step (default 10);
further arguments pass through to `code/render_sample_cloud.py`. The frames are
written to `outputs/frames/square_icem_seed0_t0_iter{0,1,2}.png`; set `OUT_DIR`
to override the directory.

Render a saved run as an RTX path-traced video:

```bash
uv run --project . python code/rendering.py --run-path data/square_icem_cylinder.npz \
  --object-shape cylinder --output outputs/square_icem_cylinder.mp4 --gif
```

The renderer reads the saved trajectory without re-simulating and adds a
decorative Franka FR3 whose end effector follows the arm base; `--franka-base`
moves it. The numbered PNG frames are kept under `outputs/<stem>_frames/frames/`.
The committed videos, `outputs/square_icem_{seed0,box100g,cylinder}.{mp4,gif}`,
are stored in Git LFS.
Panel C reads frames 75, 189, 318 and 453 directly from
`outputs/square_icem_seed0_frames/frames/`, so render `data/square_icem_seed0.npz`
with `--output outputs/square_icem_seed0.mp4` first. Encoding the MP4 and GIF
requires ffmpeg.
