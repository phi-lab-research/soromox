"""Check registry requirements that Twine's metadata validation does not enforce."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement


def test_published_dependencies_do_not_use_direct_urls():
    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )["project"]
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    direct_urls = [item for item in requirements if Requirement(item).url is not None]
    assert not direct_urls, f"PyPI rejects direct URL dependencies: {direct_urls}"
