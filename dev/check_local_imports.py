"""check_local_imports.py -- catch a Python trap that crashes a whole generator at its first step.

An `import x as y` inside a function makes `y` a LOCAL name for the ENTIRE function. If the same function uses `y`
anywhere ABOVE that import, it fails with UnboundLocalError. In gantry_robot._g_camera (which has a local
`import pybullet as p` inside its loop) that took down every agent at once. This scans the project's own modules
for the pattern.

  python dev/check_local_imports.py
"""
import ast, glob, os, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
bad = []
for path in sorted(glob.glob(os.path.join(ROOT, "*.py")) + glob.glob(os.path.join(ROOT, "pybullet_sim", "*.py"))):
    tree = ast.parse(open(path, encoding="utf-8").read())
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        first_import = {}
        for n in ast.walk(fn):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    first_import.setdefault((a.asname or a.name).split(".")[0], n.lineno)
        for name, line in first_import.items():
            early = sorted(x.lineno for x in ast.walk(fn) if isinstance(x, ast.Name) and x.id == name and x.lineno < line)
            if early:
                bad.append(f"{os.path.relpath(path, ROOT)}: {fn.name}() uses '{name}' at line {early[0]} "
                           f"before its local import at line {line}")
print("\n".join(bad) if bad else "OK: no function uses a name above a local import of the same name")
sys.exit(1 if bad else 0)
