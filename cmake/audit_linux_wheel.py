"""Reject non-relocatable native-library paths in a Linux YE3T wheel."""

from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile


NATIVE_MODULE_FRAGMENTS = (
    "ye3t/core/_factorized_runtime_native",
    "ye3t/representations/_permutation_subduction_native",
    "ye3t/runtime/_execution_plan_native",
)
TORCH_RELATIVE_RPATH = "$ORIGIN/../../torch/lib"


def dynamic_paths(dynamic_section):
    paths = []
    for line in dynamic_section.splitlines():
        if "(RPATH)" not in line and "(RUNPATH)" not in line:
            continue
        match = re.search(r"Library r(?:un)?path: \[([^\]]*)\]", line)
        if match is None:
            raise RuntimeError(f"could not parse dynamic path line: {line}")
        paths.extend(value for value in match.group(1).split(":") if value)
    return tuple(paths)


def validate_dynamic_paths(module_name, paths):
    absolute = tuple(value for value in paths if Path(value).is_absolute())
    if absolute:
        raise RuntimeError(
            f"{module_name} embeds absolute runtime paths: {absolute}"
        )
    if TORCH_RELATIVE_RPATH not in paths:
        raise RuntimeError(
            f"{module_name} does not include {TORCH_RELATIVE_RPATH}"
        )


def audit_wheel(wheel_path):
    wheel_path = Path(wheel_path).resolve()
    if not wheel_path.is_file():
        raise FileNotFoundError(wheel_path)
    with TemporaryDirectory(prefix="ye3t-wheel-audit-") as temporary:
        with ZipFile(wheel_path) as archive:
            names = tuple(
                name
                for name in archive.namelist()
                if name.endswith(".so")
                and any(
                    fragment in name
                    for fragment in NATIVE_MODULE_FRAGMENTS
                )
            )
            if len(names) != len(NATIVE_MODULE_FRAGMENTS):
                raise RuntimeError(
                    "wheel does not contain all three YE3T native modules: "
                    f"{names}"
                )
            archive.extractall(temporary, names)
        for name in names:
            module_path = Path(temporary) / name
            result = subprocess.run(
                ("readelf", "-d", str(module_path)),
                check=True,
                capture_output=True,
                text=True,
            )
            validate_dynamic_paths(name, dynamic_paths(result.stdout))
    print(f"YE3T Linux wheel dependency audit passed: {wheel_path}")


def main():
    if sys.platform != "linux":
        raise RuntimeError("audit_linux_wheel.py requires Linux and readelf")
    if len(sys.argv) != 2:
        raise RuntimeError("usage: python cmake/audit_linux_wheel.py WHEEL")
    audit_wheel(sys.argv[1])


if __name__ == "__main__":
    main()
