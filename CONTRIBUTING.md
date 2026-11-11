# Contributing to SoRoMoX

Thank you for helping improve SoRoMoX. This file covers the shortest path to a
working development checkout; the
[complete contributor guide](https://phi-lab-research.github.io/soromox/development/contributing/)
is the canonical reference for project conventions and workflows.

## Set up an editable checkout

```bash
git clone https://github.com/YOUR_USERNAME/soromox.git
cd soromox
python -m pip install -e ".[dev,docs,examples]"
pre-commit install
```

Using uv instead:

```bash
uv sync --extra dev --extra docs --extra examples
uv run pre-commit install
```

## Validate a change

Run the checks relevant to your change before opening a pull request:

```bash
uv run pytest
uv run ruff format --check src tests examples
uv run ruff check src tests examples
uv run --extra docs zensical build --clean
```

On macOS, if importing Open3D and PyTorch together causes `OMP: Error #15`,
follow the [OpenMP runtime setup](third_party/open3d/README.md#macos-openmp-runtimes)
before running tests.

See the [detailed contributor guide](docs/development/contributing.md) for code
style, tests and coverage, documentation authoring, extension points, and the
pull-request process.
