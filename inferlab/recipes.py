"""Run recipes.

`configs/default.yaml` is the only place a measured configuration is declared. Loading it here
(rather than spreading defaults across CLI flags) is what makes `inferlab bench --config
vllm-nvfp4` reproduce a specific row in docs/RESULTS.md instead of "something like last time".
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_RECIPES = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_RECIPES
    if not p.exists():
        raise FileNotFoundError(f"recipe file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    data.setdefault("defaults", {})
    data.setdefault("configs", {})
    return data


def names(path: str | Path | None = None) -> list[str]:
    return sorted(load(path).get("configs", {}))


def get(name: str, path: str | Path | None = None) -> dict[str, Any]:
    """Merge a named config over `defaults`. Unknown names raise with the available list."""
    data = load(path)
    configs = data.get("configs", {})
    if name not in configs:
        raise KeyError(
            f"unknown config {name!r}; available: {', '.join(sorted(configs)) or '(none)'}"
        )
    merged = dict(data.get("defaults", {}))
    merged.update(configs[name] or {})
    merged["name"] = name
    return merged


def describe(name: str, path: str | Path | None = None) -> str:
    cfg = get(name, path)
    bits = [f"[{name}] engine={cfg.get('engine', 'unknown')}"]
    if cfg.get("description"):
        bits.append(cfg["description"])
    if cfg.get("notes"):
        bits.append(cfg["notes"].strip().replace("\n", " "))
    return " -- ".join(bits)
