#!/usr/bin/env python3
"""
Extract all docstrings from a Python codebase without importing modules.

Usage:
  python extract_docstrings.py --root "C:/Users/lenovo/Agent" --out docstrings.md
  python extract_docstrings.py --root ./Agent --out docstrings.json --format json
  python extract_docstrings.py --root ./Agent --include-private
"""

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

SKIP_DIRS = {
    "__pycache__", ".git", ".hg", ".svn", ".venv", "venv", "env", ".mypy_cache",
    ".pytest_cache", ".ipynb_checkpoints", ".idea", ".vscode", "site-packages",
}

def should_skip_dir(p: Path) -> bool:
    name = p.name
    return name in SKIP_DIRS or name.startswith("build") or name.startswith("dist")

def read_text(path: Path) -> Optional[str]:
    # Try utf-8 first, then fall back
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="strict")
        except Exception:
            continue
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return None

def node_docstring(node) -> Optional[str]:
    ds = ast.get_docstring(node, clean=True)
    return ds if ds and ds.strip() else None

def format_location(file: Path, node: ast.AST) -> str:
    lineno = getattr(node, "lineno", None)
    endlineno = getattr(node, "end_lineno", None)
    if lineno and endlineno:
        return f"{file}:{lineno}-{endlineno}"
    if lineno:
        return f"{file}:{lineno}"
    return str(file)

def collect_docstrings_from_file(pyfile: Path, include_private: bool) -> Dict:
    src = read_text(pyfile)
    if src is None:
        return {"file": str(pyfile), "error": "Unable to read file", "items": []}

    try:
        tree = ast.parse(src, filename=str(pyfile))
    except SyntaxError as e:
        return {"file": str(pyfile), "error": f"SyntaxError: {e}", "items": []}

    out = []
    # Module-level docstring
    mod_ds = node_docstring(tree)
    if mod_ds:
        out.append({
            "kind": "module",
            "name": pyfile.stem,
            "qualname": pyfile.stem,
            "location": format_location(pyfile, tree),
            "doc": mod_ds,
        })

    # Traverse everything to catch classes/functions/methods, including nested
    class StackVisitor(ast.NodeVisitor):
        def __init__(self):
            self.stack: List[str] = []

        def _include(self, name: str) -> bool:
            if include_private:
                return True
            return not name.startswith("_")

        def visit_ClassDef(self, node: ast.ClassDef):
            if self._include(node.name):
                qual = ".".join(self.stack + [node.name]) if self.stack else node.name
                ds = node_docstring(node)
                out.append({
                    "kind": "class",
                    "name": node.name,
                    "qualname": qual,
                    "location": format_location(pyfile, node),
                    "doc": ds,
                })
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            if self._include(node.name):
                qual = ".".join(self.stack + [node.name]) if self.stack else node.name
                ds = node_docstring(node)
                out.append({
                    "kind": "method" if any(isinstance(p, ast.ClassDef) for p in self._parents(node)) else "function",
                    "name": node.name,
                    "qualname": qual,
                    "location": format_location(pyfile, node),
                    "doc": ds,
                })
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            # Treat async functions similarly
            if self._include(node.name):
                qual = ".".join(self.stack + [node.name]) if self.stack else node.name
                ds = node_docstring(node)
                out.append({
                    "kind": "method" if any(isinstance(p, ast.ClassDef) for p in self._parents(node)) else "function",
                    "name": node.name,
                    "qualname": qual,
                    "location": format_location(pyfile, node),
                    "doc": ds,
                })
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        # Helper to detect if function is inside a class
        def _parents(self, node):
            # ast doesn't give parents; recompute by walking (cheap enough per call).
            parents = []
            def walk(n, stack):
                for child in ast.iter_child_nodes(n):
                    stack.append(n)
                    if child is node:
                        parents.extend(stack)
                        return True
                    if walk(child, stack):
                        return True
                    stack.pop()
                return False
            walk(tree, [])
            return parents

    StackVisitor().visit(tree)
    return {"file": str(pyfile), "items": out}

def walk_python_files(root: Path) -> List[Path]:
    pyfiles: List[Path] = []
    for base, dirs, files in os.walk(root):
        # Prune skip dirs
        dirs[:] = [d for d in dirs if not should_skip_dir(Path(base) / d)]
        for f in files:
            if f.endswith(".py"):
                pyfiles.append(Path(base) / f)
    return sorted(pyfiles)

def render_markdown(results: List[Dict], root: Path, relative: bool) -> str:
    lines = ["# Docstring Index\n"]
    for file_result in results:
        file_path = Path(file_result["file"])
        title = str(file_path.relative_to(root)) if relative else str(file_path)
        err = file_result.get("error")
        lines.append(f"## {title}")
        if err:
            lines.append(f"> **Error:** {err}\n")
            continue
        if not file_result["items"]:
            lines.append("_No docstrings found._\n")
            continue
        for item in file_result["items"]:
            kind = item["kind"].capitalize()
            qual = item["qualname"]
            loc = item["location"]
            doc = item["doc"] or "_(no docstring)_"
            lines.append(f"### {kind}: `{qual}`")
            lines.append(f"- **Location:** `{loc}`")
            lines.append("")
            lines.append("```text")
            lines.append(doc)
            lines.append("```\n")
    return "\n".join(lines).strip() + "\n"

def main():
    ap = argparse.ArgumentParser(description="Extract docstrings from a Python codebase without importing.")
    ap.add_argument("--root", required=True, help="Root directory (e.g., C:/Users/lenovo/Agent)")
    ap.add_argument("--out", required=True, help="Output file path (.md or .json)")
    ap.add_argument("--format", choices=["md", "json"], help="Force output format (default: by extension)")
    ap.add_argument("--relative", action="store_true", help="Show file paths relative to --root in Markdown")
    ap.add_argument("--include-private", action="store_true", help="Include names starting with underscore")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    fmt = args.format or ("json" if out.suffix.lower() == ".json" else "md")

    pyfiles = walk_python_files(root)
    results = [collect_docstrings_from_file(p, include_private=args.include_private) for p in pyfiles]

    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        out.write_text(json.dumps({"root": str(root), "results": results}, indent=2), encoding="utf-8")
    else:
        md = render_markdown(results, root, relative=args.relative)
        out.write_text(md, encoding="utf-8")

    print(f"[OK] Wrote {out}")

if __name__ == "__main__":
    main()
