import os
import sys

from ye3t.runtime.triton_config import configure_triton_c_compiler


def test_triton_compiler_config_finds_conda_wrapper_from_python_executable(tmp_path, monkeypatch):
    env_bin = tmp_path / "env" / "bin"
    env_bin.mkdir(parents=True)
    python_exe = env_bin / "python"
    python_exe.write_text("#!/bin/sh\n")
    python_exe.chmod(0o755)
    compiler = env_bin / "x86_64-conda-linux-gnu-gcc"
    compiler.write_text("#!/bin/sh\n")
    compiler.chmod(0o755)
    cxx = env_bin / "x86_64-conda-linux-gnu-g++"
    cxx.write_text("#!/bin/sh\n")
    cxx.chmod(0o755)
    nvcc = env_bin / "nvcc"
    nvcc.write_text("#!/bin/sh\n")
    nvcc.chmod(0o755)

    monkeypatch.delenv("CC", raising=False)
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(sys, "executable", str(python_exe))
    monkeypatch.setattr("ye3t.runtime.triton_config.shutil.which", lambda _name: None)

    selected = configure_triton_c_compiler(required=True)

    assert selected == str(compiler)
    assert os.environ["CC"] == str(compiler)
    assert os.environ["CXX"] == str(cxx)
    assert os.environ["CUDA_HOME"] == str(env_bin.parent)
    assert os.environ["CUDA_PATH"] == str(env_bin.parent)
