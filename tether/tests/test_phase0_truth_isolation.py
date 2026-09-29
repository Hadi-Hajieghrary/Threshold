"""Phase 0 truth-isolation acceptance tests."""

from pathlib import Path

from tether.estimation.truth_isolation import lint_all_arms, lint_source


def test_p0_t5_arm_imports_are_isolated_or_declared():
    assert lint_all_arms() == {"arms": [], "oracle": []}


def test_p0_t5_undeclared_oracle_is_rejected():
    oracle_path = Path(__file__).parents[1] / "estimation" / "oracle.py"
    source = oracle_path.read_text(encoding="ascii")
    undeclared_source = source.replace(
        'TRUTH_ISOLATION_EXEMPTIONS = {"tether.physics.plant"}',
        "TRUTH_ISOLATION_EXEMPTIONS = set()",
    )
    violations = lint_source(undeclared_source, "undeclared_oracle")
    assert violations
    assert "tether.physics.plant" in violations[0]