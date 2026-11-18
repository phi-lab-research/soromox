# Paper experiments

The **[SoRoMoX paper experiments](https://github.com/phi-lab-research/soromox-paper-experiments)**
repository provides the experiment code, datasets, checkpoints, figures, and
reproduction tests for the paper.

Start with its [experiment index](https://github.com/phi-lab-research/soromox-paper-experiments/blob/main/paper_results/README.md)
and [setup instructions](https://github.com/phi-lab-research/soromox-paper-experiments#setup).
The experiments use a pinned SoRoMoX release and include checksums for the
canonical data and outputs.

```bash
git clone https://github.com/phi-lab-research/soromox-paper-experiments.git
cd soromox-paper-experiments
git lfs pull
uv sync --locked --group test --group paper-benchmarks
```
