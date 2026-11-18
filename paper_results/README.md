# Paper experiments moved

The experiment code, datasets, checkpoints, figures, and reproduction tests now
live in **[phi-lab-research/soromox-paper-experiments](https://github.com/phi-lab-research/soromox-paper-experiments)**.

Start with the [experiment index](https://github.com/phi-lab-research/soromox-paper-experiments/blob/main/paper_results/README.md)
and [setup instructions](https://github.com/phi-lab-research/soromox-paper-experiments#setup).
The new repository installs a pinned SoRoMoX release and retains the relevant
file history, source revisions, and checksums of the canonical paper artifacts.

```bash
git clone https://github.com/phi-lab-research/soromox-paper-experiments.git
cd soromox-paper-experiments
git lfs pull
uv sync --locked --group test --group paper-benchmarks
```

Historical SoRoMoX tags and commits still contain the original experiment files.
The migration source was [b4856ec7](https://github.com/phi-lab-research/soromox/tree/b4856ec7c4ca5c6509255945f004aa7516bbe114/paper_results).
The old `soromox[paper_results]` extra is deprecated and now an empty compatibility
alias; install the experiment repository's environment instead.
