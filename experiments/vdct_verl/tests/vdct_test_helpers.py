"""Shared test helpers: script loading and JSONL fixtures.

The scripts are standalone executables, so tests load them by file path;
this module keeps that loader (and the fixture writer) in one place.
"""

import importlib.util
import json
from pathlib import Path

VDCT_VERL = Path(__file__).resolve().parents[1]


def load_script(name: str):
    """Import ``scripts/<name>.py`` as a module, by file path."""
    spec = importlib.util.spec_from_file_location(name, VDCT_VERL / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path
