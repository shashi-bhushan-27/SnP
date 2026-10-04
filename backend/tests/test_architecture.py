"""Enforces the module boundaries described in docs/architecture.md.

    core                      imports nothing else from prism
    nlp, etl, storage, modules  may import prism.core only (modules never import each other's layers)
    api, bootstrap, cli, config  the composition root / entry points: unrestricted

If this fails, a layer reached into something it should receive through a protocol instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PRISM = Path(__file__).resolve().parents[1] / "prism"
ALLOWED: dict[str, set[str]] = {
    "core": set(),
    "nlp": {"core"},
    "etl": {"core"},
    "storage": {"core"},
    "modules": {"core"},
}


def _prism_imports(path: Path) -> set[str]:
    module = ".".join(path.relative_to(PRISM.parent).with_suffix("").parts)
    package = module.rsplit(".", 1)[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - (node.level - 1)]
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            found.add(target)
            found.update(f"{target}.{alias.name}" for alias in node.names)
    return {m for m in found if m.startswith("prism.")}


def _layer(dotted: str) -> str:
    return dotted.split(".")[1]


def _files(layer: str) -> list[Path]:
    return sorted((PRISM / layer).rglob("*.py"))


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_layer_only_imports_allowed_layers(layer: str) -> None:
    violations = []
    for path in _files(layer):
        for imported in _prism_imports(path):
            target = _layer(imported)
            if target.startswith("_") or target == layer:
                continue
            if target not in ALLOWED[layer]:
                violations.append(f"{path.relative_to(PRISM.parent)} imports {imported}")
    assert not violations, "\n".join(violations)


def test_every_layer_directory_is_covered() -> None:
    packages = {p.name for p in PRISM.iterdir() if p.is_dir() and not p.name.startswith("_")}
    assert packages <= set(ALLOWED) | {"api"}, f"unclassified package(s): {packages - set(ALLOWED) - {'api'}}"
