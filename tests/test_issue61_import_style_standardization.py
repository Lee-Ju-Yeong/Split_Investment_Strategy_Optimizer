import ast
from pathlib import Path
import unittest


class TestIssue61ImportStyleStandardization(unittest.TestCase):
    def test_src_internal_imports_are_relative(self):
        repo_root = Path(__file__).resolve().parent.parent
        src_dir = repo_root / "src"

        internal = {p.stem for p in src_dir.glob("*.py") if p.stem != "__init__"}

        violations: list[str] = []
        for path in sorted(src_dir.glob("*.py")):
            if path.name == "__init__.py":
                continue

            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if not node.module or node.level:
                        continue

                    # Within src/, importing via `src.*` is discouraged; prefer relative imports.
                    if node.module == "src" or node.module.startswith("src."):
                        violations.append(f"{path}:{node.lineno}: from {node.module} import ...")
                        continue

                    base = node.module.split(".", 1)[0]
                    if base in internal:
                        violations.append(f"{path}:{node.lineno}: from {node.module} import ...")

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        base = alias.name.split(".", 1)[0]
                        if base in internal:
                            violations.append(f"{path}:{node.lineno}: import {alias.name}")

        self.assertEqual(
            violations,
            [],
            msg="Non-relative imports found inside src/. Use `from .<module> import ...`.\n"
            + "\n".join(violations),
        )

    def test_entrypoints_have_direct_execution_bootstrap(self):
        repo_root = Path(__file__).resolve().parent.parent
        src_dir = repo_root / "src"

        missing: list[str] = []
        misplaced: list[str] = []

        for path in sorted(src_dir.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            if "if __name__ == \"__main__\":" not in text and "if __name__ == '__main__':" not in text:
                continue

            lines = text.splitlines()
            bootstrap_idx = next((i for i, line in enumerate(lines) if "# BOOTSTRAP:" in line), None)
            if bootstrap_idx is None:
                missing.append(str(path))
                continue

            first_relative_import_idx = next(
                (i for i, line in enumerate(lines) if line.lstrip().startswith("from .")),
                None,
            )
            if first_relative_import_idx is not None and bootstrap_idx > first_relative_import_idx:
                misplaced.append(
                    f"{path}: bootstrap line {bootstrap_idx + 1} occurs after first relative import line {first_relative_import_idx + 1}"
                )

        self.assertEqual(
            missing,
            [],
            msg="Entrypoint modules missing direct-execution bootstrap.\n" + "\n".join(missing),
        )
        self.assertEqual(
            misplaced,
            [],
            msg="Bootstrap must appear before relative imports.\n" + "\n".join(misplaced),
        )

