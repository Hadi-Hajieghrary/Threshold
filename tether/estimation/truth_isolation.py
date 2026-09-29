"""AST enforcement of estimation-to-plant truth isolation."""

from __future__ import annotations

import ast
from pathlib import Path

PLANT_SIDE_MODULES = ("tether.physics.plant", "tether.physics.weather")
EXEMPTION_NAME = "TRUTH_ISOLATION_EXEMPTIONS"


def _declared_exemptions(tree: ast.AST) -> set[str]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == EXEMPTION_NAME for target in targets):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    return set()
                return set(value)
    return set()


def lint_source(source: str, module_name: str) -> list[str]:
    """Return undeclared imports from modules designated as plant-side."""
    tree = ast.parse(source, filename=module_name)
    exemptions = _declared_exemptions(tree)
    violations = []
    for node in ast.walk(tree):
        imported_modules = []
        if isinstance(node, ast.Import):
            imported_modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules = [node.module]
        for imported_module in imported_modules:
            is_plant_side = any(
                imported_module == prefix or imported_module.startswith(prefix + ".")
                for prefix in PLANT_SIDE_MODULES
            )
            is_exempt = any(
                imported_module == exemption or imported_module.startswith(exemption + ".")
                for exemption in exemptions
            )
            if is_plant_side and not is_exempt:
                violations.append(
                    f"{module_name}:{node.lineno}: undeclared plant-side import {imported_module}"
                )
    return violations


def lint_module(path: Path) -> list[str]:
    return lint_source(path.read_text(encoding="ascii"), path.as_posix())


def lint_all_arms() -> dict[str, list[str]]:
    package = Path(__file__).parent
    return {
        name: lint_module(package / f"{name}.py")
        for name in ("arms", "oracle")
    }


PLANT_SIDE_SYSTEM_NAMES = frozenset({"force_adder"})


def is_plant_side(system) -> bool:
    """A system is plant-side by class declaration, or as the plant, scene graph, or adder."""
    from pydrake.geometry import SceneGraph
    from pydrake.multibody.plant import MultibodyPlant

    if isinstance(system, (MultibodyPlant, SceneGraph)):
        return True
    if getattr(type(system), "PLANT_SIDE", False):
        return True
    return system.get_name() in PLANT_SIDE_SYSTEM_NAMES


def lint_diagram(diagram, exemptions: set[str] | frozenset[str] = frozenset()) -> list[str]:
    """Walk the built diagram's connection map (IV.11).

    Fails any input of a non-plant-side system fed by a plant-side output, except the
    sanctioned measurement outputs of sensor suites and systems named in ``exemptions``
    (the oracle arm's declared exemption).
    """
    violations = []
    for (input_system, input_index), (output_system, output_index) in diagram.connection_map().items():
        if is_plant_side(input_system):
            continue
        if not is_plant_side(output_system):
            continue
        if getattr(type(output_system), "SANCTIONED_OUTPUTS", False):
            continue
        if input_system.get_name() in exemptions:
            continue
        violations.append(
            f"{input_system.get_name()}[{input_system.get_input_port(int(input_index)).get_name()}]"
            f" <- {output_system.get_name()}[{output_system.get_output_port(int(output_index)).get_name()}]"
        )
    return violations