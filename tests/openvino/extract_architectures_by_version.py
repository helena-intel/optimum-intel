"""
Extract supported architectures per transformers version from test_genai.py.

Reloads the test_genai module with a mocked is_transformers_version for each
target version, then reads the class-level ALL_SUPPORTED_ARCHITECTURES tuples.

The version list is built automatically from min/max bounds in setup.py
(the transformers requirement specifier), iterating over each minor version.
"""

# ruff: noqa: I001  # Avoid black/ruff conflict

import importlib
import re
from pathlib import Path
from unittest.mock import patch

from packaging.specifiers import SpecifierSet
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_transformers_specifier():
    """Extract the transformers version specifier from setup.py (e.g. '>=4.45,<5.1')."""
    setup_py = REPO_ROOT / "setup.py"
    for line in setup_py.read_text().splitlines():
        m = re.search(r'"transformers([><=!~][^"]+)"', line)
        if m:
            return SpecifierSet(m.group(1))
    raise RuntimeError("Could not find transformers requirement in setup.py")


def _build_transformers_versions():
    """Build sorted list of versions to test: one per minor version within setup.py bounds.

    Uses patch version 99 for each minor to represent "latest patch of that minor".
    For each major version, iterates minors from the lower bound up to the highest
    minor referenced in test_genai.py thresholds for that major, then advances.
    """
    spec = _parse_transformers_specifier()

    min_version = None
    for s in spec:
        if ">=" in s.operator or ">" in s.operator:
            min_version = Version(s.version)

    # Find the max minor per major referenced in test_genai.py
    test_genai_path = Path(__file__).with_name("test_genai.py")
    text = test_genai_path.read_text()
    threshold_strs = re.findall(r'is_transformers_version\("[><=!]+",\s*"([^"]+)"\)', text)
    max_minor_per_major = {}
    for v_str in threshold_strs:
        v = Version(v_str)
        max_minor_per_major[v.major] = max(max_minor_per_major.get(v.major, 0), v.minor)

    versions = []
    major, minor = min_version.major, min_version.minor
    while True:
        v = Version(f"{major}.{minor}.99")
        if v not in spec:
            break
        versions.append(v)
        minor += 1
        if minor > max_minor_per_major.get(major, 0):
            major += 1
            minor = 0

    return versions


def get_architectures_by_version():
    """Build dict of {version: {"llm": set(...), "vlm": set(...)}} by reloading test_genai."""
    import test_genai

    versions = _build_transformers_versions()
    result = {}
    for version in versions:
        with patch("optimum.utils.import_utils._transformers_version", str(version)):
            importlib.reload(test_genai)
            result[str(version)] = {
                "llm": set(test_genai.LLMPipelineTestCase.ALL_SUPPORTED_ARCHITECTURES),
                "vlm": set(test_genai.VLMPipelineTestCase.ALL_SUPPORTED_ARCHITECTURES),
            }

    return result


def deduplicate(archs_by_version):
    """Remove consecutive versions with identical architectures, always keeping the last."""
    versions = list(archs_by_version.keys())
    keep = set()
    keep.add(versions[-1])
    for i in range(len(versions) - 2, -1, -1):
        if archs_by_version[versions[i]] != archs_by_version[versions[i + 1]]:
            keep.add(versions[i])
    return {v: archs_by_version[v] for v in versions if v in keep}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-versions",
        action="store_true",
        help="Output version list as JSON array (lightweight, no heavy deps needed)",
    )
    parser.add_argument(
        "--deduplicated",
        action="store_true",
        help="Only include versions with distinct architecture sets (requires full deps)",
    )
    args = parser.parse_args()

    if args.json_versions:
        archs_by_version = deduplicate(get_architectures_by_version())
        # Convert "4.51.99" -> "4.51.*" for use with pip install
        versions = [v.rsplit(".", 1)[0] + ".*" for v in archs_by_version.keys()]
        print(json.dumps(versions))
    elif args.deduplicated:
        archs_by_version = deduplicate(get_architectures_by_version())
        for version, archs in archs_by_version.items():
            print(f"\nTransformers {version}:")
            print(f"  LLM ({len(archs['llm'])}): {sorted(archs['llm'])}")
            print(f"  VLM ({len(archs['vlm'])}): {sorted(archs['vlm'])}")
    else:
        archs_by_version = get_architectures_by_version()
        for version, archs in archs_by_version.items():
            print(f"\nTransformers {version}:")
            print(f"  LLM ({len(archs['llm'])}): {sorted(archs['llm'])}")
            print(f"  VLM ({len(archs['vlm'])}): {sorted(archs['vlm'])}")
