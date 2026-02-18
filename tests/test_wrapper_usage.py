import ast
import unittest
from pathlib import Path


FORBIDDEN_WRAPPER_MODULES = {
    "backtester",
    "strategy",
    "portfolio",
    "execution",
    "backtest_strategy_gpu",
    "daily_stock_tier_batch",
    "financial_collector",
    "investor_trading_collector",
}

FORBIDDEN_WRAPPER_FILES = {f"{name}.py" for name in FORBIDDEN_WRAPPER_MODULES}
FORBIDDEN_ABSOLUTE_MODULES = {f"src.{name}" for name in FORBIDDEN_WRAPPER_MODULES}
SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def _is_forbidden_absolute_module_name(module_name: str) -> bool:
    if not module_name:
        return False
    if module_name in FORBIDDEN_WRAPPER_MODULES:
        return True
    if module_name in FORBIDDEN_ABSOLUTE_MODULES:
        return True
    return False


def _resolve_import_from_module(path: Path, node: ast.ImportFrom) -> str:
    module = node.module or ""
    if node.level == 0:
        return module

    rel_parts = path.relative_to(SRC_DIR).with_suffix("").parts
    current_package_parts = ["src", *rel_parts[:-1]]

    keep_count = len(current_package_parts) - (node.level - 1)
    base_parts = current_package_parts[: max(0, keep_count)]
    if module:
        base_parts.extend(module.split("."))
    return ".".join(base_parts)


def _resolve_relative_alias_module(path: Path, level: int, alias_name: str) -> str:
    rel_parts = path.relative_to(SRC_DIR).with_suffix("").parts
    current_package_parts = ["src", *rel_parts[:-1]]
    keep_count = len(current_package_parts) - (level - 1)
    base_parts = current_package_parts[: max(0, keep_count)]
    base_parts.append(alias_name)
    return ".".join(base_parts)


def _iter_runtime_module_files():
    for path in sorted(SRC_DIR.rglob("*.py")):
        yield path


def _find_forbidden_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    findings = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported = alias.name
                if _is_forbidden_absolute_module_name(imported):
                    findings.append((node.lineno, f"import {imported}"))

        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            resolved_module = _resolve_import_from_module(path, node)

            if _is_forbidden_absolute_module_name(resolved_module):
                findings.append((node.lineno, f"from {resolved_module} import ..."))

            # from src import strategy
            if node.level == 0 and module == "src":
                for alias in node.names:
                    if alias.name in FORBIDDEN_WRAPPER_MODULES:
                        findings.append((node.lineno, f"from src import {alias.name}"))

            # from . import strategy
            if node.level > 0 and not module:
                for alias in node.names:
                    resolved_alias = _resolve_relative_alias_module(path, node.level, alias.name)
                    if _is_forbidden_absolute_module_name(resolved_alias):
                        findings.append((node.lineno, f"from . import {alias.name}"))

    return findings


class TestWrapperUsageGuard(unittest.TestCase):
    def test_removed_wrapper_files_must_not_exist(self):
        violations = []
        repo_root = Path(__file__).resolve().parent.parent
        for filename in sorted(FORBIDDEN_WRAPPER_FILES):
            wrapper_path = SRC_DIR / filename
            if wrapper_path.exists():
                violations.append(str(wrapper_path.relative_to(repo_root)))

        self.assertFalse(
            violations,
            "Removed wrapper files should not exist:\n" + "\n".join(violations),
        )

    def test_runtime_code_must_not_import_conditional_wrappers(self):
        violations = []
        for path in _iter_runtime_module_files():
            for lineno, statement in _find_forbidden_imports(path):
                rel_path = path.relative_to(Path(__file__).resolve().parent.parent)
                violations.append(f"{rel_path}:{lineno} -> {statement}")

        self.assertFalse(
            violations,
            "Conditional wrapper imports detected in runtime modules:\n"
            + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
