
"""Benchmark helpers for exact primitive quotients and timeout behavior."""
import multiprocessing as mp
from time import perf_counter
from ye3t._record import recordclass


@recordclass(('nin', 'lin', 'L_R', 'mode', 'timeout_seconds', 'status', 'sector_dim', 'primitive_rank', 'sector_seconds', 'quotient_seconds', 'detail'), frozen = True)
class ExactPrimitiveBenchmarkResult:
    detail = ""


def _exact_benchmark_worker(queue, args):
    try:
        nin, lin, L_R, mode, tree_type = args
        from ye3t.core.product_engine import ExactProductExpansionEngine

        engine = ExactProductExpansionEngine(tree_type=tree_type)
        t0 = perf_counter()
        sector = engine.sector(nin, lin, L_R)
        t1 = perf_counter()
        quotient = engine.primitive_quotient(nin, lin, L_R, mode=mode)
        t2 = perf_counter()
        queue.put(("ok", {
            "sector_dim": int(sector.dim),
            "primitive_rank": int(quotient.primitive_rank),
            "sector_seconds": float(t1 - t0),
            "quotient_seconds": float(t2 - t1),
        }))
    except Exception as exc:  # pragma: no cover - worker exception path
        queue.put(("error", repr(exc)))


def benchmark_exact_primitive_case(
    nin,
    lin,
    L_R,
    *,
    mode,
    timeout_seconds = 30.0,
    tree_type = "balanced",
):
    """Benchmark one exact primitive quotient with a hard timeout."""
    args = (tuple(int(x) for x in nin), tuple(int(x) for x in lin), int(L_R), str(mode), str(tree_type))
    available = set(mp.get_all_start_methods())
    if "fork" not in available:
        from ye3t.core.product_engine import ExactProductExpansionEngine

        try:
            engine = ExactProductExpansionEngine(tree_type=tree_type)
            t0 = perf_counter()
            sector = engine.sector(args[0], args[1], args[2])
            t1 = perf_counter()
            quotient = engine.primitive_quotient(args[0], args[1], args[2], mode=str(mode))
            t2 = perf_counter()
            return ExactPrimitiveBenchmarkResult(
                nin=args[0],
                lin=args[1],
                L_R=args[2],
                mode=str(mode),
                timeout_seconds=float(timeout_seconds),
                status="exact",
                sector_dim=int(sector.dim),
                primitive_rank=int(quotient.primitive_rank),
                sector_seconds=float(t1 - t0),
                quotient_seconds=float(t2 - t1),
                detail="",
            )
        except Exception as exc:
            return ExactPrimitiveBenchmarkResult(
                nin=args[0],
                lin=args[1],
                L_R=args[2],
                mode=str(mode),
                timeout_seconds=float(timeout_seconds),
                status="error",
                sector_dim=None,
                primitive_rank=None,
                sector_seconds=None,
                quotient_seconds=None,
                detail=repr(exc),
            )
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_exact_benchmark_worker, args=(queue, args))
    proc.start()
    proc.join(float(timeout_seconds))
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return ExactPrimitiveBenchmarkResult(
            nin=args[0],
            lin=args[1],
            L_R=args[2],
            mode=str(mode),
            timeout_seconds=float(timeout_seconds),
            status="timeout",
            sector_dim=None,
            primitive_rank=None,
            sector_seconds=None,
            quotient_seconds=None,
            detail=f"exact quotient exceeded {float(timeout_seconds):.1f} s budget",
        )
    if queue.empty():
        return ExactPrimitiveBenchmarkResult(
            nin=args[0],
            lin=args[1],
            L_R=args[2],
            mode=str(mode),
            timeout_seconds=float(timeout_seconds),
            status="error",
            sector_dim=None,
            primitive_rank=None,
            sector_seconds=None,
            quotient_seconds=None,
            detail="worker exited without returning a result",
        )
    status, payload = queue.get()
    if status == "ok":
        return ExactPrimitiveBenchmarkResult(
            nin=args[0],
            lin=args[1],
            L_R=args[2],
            mode=str(mode),
            timeout_seconds=float(timeout_seconds),
            status="exact",
            sector_dim=payload["sector_dim"],
            primitive_rank=payload["primitive_rank"],
            sector_seconds=payload["sector_seconds"],
            quotient_seconds=payload["quotient_seconds"],
            detail="",
        )
    return ExactPrimitiveBenchmarkResult(
        nin=args[0],
        lin=args[1],
        L_R=args[2],
        mode=str(mode),
        timeout_seconds=float(timeout_seconds),
        status="error",
        sector_dim=None,
        primitive_rank=None,
        sector_seconds=None,
        quotient_seconds=None,
        detail=str(payload),
    )


def benchmark_exact_primitive_suite(
    cases,
    *,
    timeout_seconds = 30.0,
    tree_type = "balanced",
):
    """Run a suite of exact primitive quotient benchmarks."""
    return [
        benchmark_exact_primitive_case(
            case["nin"],
            case["lin"],
            case["L_R"],
            mode=case["mode"],
            timeout_seconds=timeout_seconds,
            tree_type=tree_type,
        )
        for case in cases
    ]
