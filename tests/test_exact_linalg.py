def test_exact_algebraic_rank_handles_supported_radicals_without_sympy():
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.exact_linalg import exact_algebraic_rank_or_none
from ye3t.exact_scalars import ExactRadical

root_two = ExactRadical.sqrt(2)
root_three = ExactRadical.sqrt(3)
matrix = (
    (root_two, 0, root_two),
    (0, root_three, root_three),
    (0, 0, 0),
)

assert exact_algebraic_rank_or_none(matrix) == 2
assert "ye3t._optional_sympy" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_exact_algebraic_rank_reports_unsupported_division_without_sympy():
    import subprocess
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.exact_linalg import exact_algebraic_rank_or_none
from ye3t.exact_scalars import ExactRadical

unsupported_pivot = ExactRadical.sqrt(2) + ExactRadical.sqrt(3)

assert exact_algebraic_rank_or_none(((unsupported_pivot,),)) is None
assert "ye3t._optional_sympy" not in sys.modules
assert "sympy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
