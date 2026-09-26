# ye3t Dependency Matrix

`ye3t` keeps optional accelerators out of the core install.

| extra | purpose | notes |
|---|---|---|
| `accelerators` | Triton, OpenEquivariance, and cuEquivariance bridges | Optional and lazy; native/PyTorch fallbacks remain the baseline. |
| `triton` | Triton kernels | Optional and lazy. |
| `oeq` | OpenEquivariance bridge | Optional and lazy. |
| `cueq` | cuEquivariance bridge | Optional and lazy. |
| `all` | all supported accelerator extras | Equivalent to `accelerators`. |
| `docs` | Sphinx documentation build | `sphinx`; install the package first because the API reference uses autodoc. |
| `reference` | exact sympy reference paths and the sympy-based examples | `sympy`; optional and lazy. |
| `dev` | test and packaging tools | `pytest`, `build`, `wheel`, `sympy`. |

`import ye3t` and `import ye3t.api` should remain lightweight and must not
eagerly import optional accelerator or ACE dependencies.
