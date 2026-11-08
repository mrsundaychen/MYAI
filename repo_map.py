"""仓库地图：目录树 + 每个 Python 文件的顶层符号（函数/类 + 行号）。

用途：快速了解仓库结构，或把地图文本注入给 agent 帮它导航（类似 Aider 的 repo map）。
可独立运行：`python3 repo_map.py [根目录]`
"""

import ast
import os
from pathlib import Path

# 跳过这些目录
IGNORE_DIRS = {
    ".git", ".venv", "__pycache__", ".idea", "node_modules",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".tox",
}
# 这些后缀的文件只列名字，不解析
IGNORE_FILE_SUFFIXES = {
    ".pyc", ".pyo", ".png", ".jpg", ".jpeg", ".gif", ".ico",
    ".pdf", ".zip", ".gz", ".tar", ".woff", ".woff2", ".ttf",
    ".sqlite", ".sqlite-shm", ".sqlite-wal", ".db",
}
# 只对源码文件提取符号
SYMBOL_SUFFIXES = {".py"}


def _py_symbols(path: Path, limit: int = 60) -> list[str]:
    """提取一个 Python 文件的顶层 def/class（带行号）。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = ", ".join(a.arg for a in node.args.args)
            out.append(f"      {node.lineno:4d} def {node.name}({args})")
        elif isinstance(node, ast.ClassDef):
            out.append(f"      {node.lineno:4d} class {node.name}")
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = ", ".join(a.arg for a in sub.args.args)
                    out.append(f"        {sub.lineno:4d} def {sub.name}({args})")
    return out[:limit]


def build_repo_map(root: str | Path) -> str:
    """生成仓库地图文本。"""
    root = Path(root).resolve()
    out = [f"{root.name}/"]
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")
        )
        rel = Path(dirpath).relative_to(root)
        depth = len(rel.parts)
        if depth > 0:
            out.append(f"{'  ' * (depth - 1)}{rel.name}/")
        for fn in sorted(filenames):
            fp = Path(dirpath) / fn
            if fn.startswith(".") or fp.suffix in IGNORE_FILE_SUFFIXES:
                continue
            out.append(f"{'  ' * depth}{fn}")
            if fp.suffix in SYMBOL_SUFFIXES:
                out.extend(_py_symbols(fp))
    return "\n".join(out)


def _local_imports(path: Path) -> list[str]:
    """提取文件的本地绝对 import 的顶层模块名（含函数内部的懒导入）。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:  # 只取绝对导入，跳过相对导入
                mods.append(node.module.split(".")[0])
    seen: set[str] = set()
    out = []
    for m in mods:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _resolve_local_module(root: Path, module: str) -> Path | None:
    """把模块名解析成项目内的文件；找不到返回 None（标准库/第三方）。"""
    for cand in (root / f"{module}.py", root / module / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def build_focused_map(root: str | Path, target: str, max_depth: int = 1) -> str:
    """按需生成「目标文件 + 它本地 import 的文件」的符号地图（实时、轻量）。

    target 用虚拟路径（如 /main.py）或相对路径。
    只解析目标文件及其本地依赖，不遍历整个仓库。
    """
    root = Path(root).resolve()
    target_path = root / target.lstrip("/")
    out: list[str] = []
    visited: set[Path] = set()

    def emit(path: Path, depth: int) -> None:
        if depth > max_depth or not path.is_file():
            return
        key = path.resolve()
        if key in visited:
            return
        visited.add(key)
        try:
            rel = path.relative_to(root)
        except ValueError:
            return
        out.append(f"{'  ' * depth}{rel}")
        out.extend(_py_symbols(path))
        for mod in _local_imports(path):
            dep = _resolve_local_module(root, mod)
            if dep:
                emit(dep, depth + 1)

    emit(target_path, 0)
    return "\n".join(out) if out else f"(未找到目标文件 {target})"


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "."
    print(build_repo_map(root))
