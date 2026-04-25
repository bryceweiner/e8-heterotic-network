"""
Driver: compute and report the E8 ⊕ E8 root-graph clustering coefficient.

Run as ``python -m e8_heterotic.cli.compute_clustering``.

The output is written to stdout via the standard logging system. A JSON
results file is written to ``results/clustering_<timestamp>.json`` with a
SHA-256 reproducibility manifest of the constructed arrays.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import platform
import sys
from typing import Any

import numpy as np

from e8_heterotic.core.adjacency import CONVENTIONS
from e8_heterotic.core.clustering import (
    count_triangles_and_wedges,
    count_triangles_networkx,
    degree_distribution,
    global_clustering_coefficient,
    mean_local_clustering_coefficient,
    num_connected_components,
)
from e8_heterotic.core.constants import (
    E8_CLUSTERING_LITERATURE_CLAIM,
    INNER_PRODUCT_TOLERANCE,
)
from e8_heterotic.core.root_system import (
    construct_e8_roots,
    construct_e8xe8_roots,
)


def _hash(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _format_int(n: int, width: int = 9) -> str:
    return f"{n:>{width},}"


def _analyze(
    roots: np.ndarray,
    label: str,
    manifest: dict[str, str],
) -> dict[str, dict[str, Any]]:
    print(f"\n{label} ({roots.shape[0]} vertices)")
    print("-" * 64)
    manifest[f"{label}_roots_sha256"] = _hash(roots)

    out: dict[str, dict[str, Any]] = {}
    for name, (descr, fn) in CONVENTIONS.items():
        adjacency = fn(roots, tolerance=INNER_PRODUCT_TOLERANCE)
        manifest[f"{label}_{name}_adjacency_sha256"] = _hash(
            np.ascontiguousarray(adjacency)
        )

        triangles, wedges = count_triangles_and_wedges(adjacency)
        triangles_nx = count_triangles_networkx(adjacency)
        if triangles != triangles_nx:
            raise RuntimeError(
                f"Triangle disagreement: trace={triangles}, nx={triangles_nx}"
            )

        c_global = global_clustering_coefficient(adjacency)
        c_local = mean_local_clustering_coefficient(adjacency)
        n_components = num_connected_components(adjacency)
        degs = degree_distribution(adjacency)
        n_edges = int(adjacency.sum()) // 2
        deg_repr = (
            int(degs[0])
            if degs.min() == degs.max()
            else f"{int(degs.min())}-{int(degs.max())}"
        )
        print(
            f"  Convention {name} ({descr:>14s}): "
            f"deg={str(deg_repr):>5s} "
            f"edges={_format_int(n_edges, 7)} "
            f"T={_format_int(triangles, 9)} "
            f"W={_format_int(wedges, 11)} "
            f"C_g={c_global:.10f} "
            f"C_lm={c_local:.10f} "
            f"comp={n_components}"
        )

        out[name] = {
            "label": descr,
            "num_vertices": int(adjacency.shape[0]),
            "num_edges": n_edges,
            "degree_min": int(degs.min()),
            "degree_max": int(degs.max()),
            "degree_mean": float(degs.mean()),
            "num_triangles": triangles,
            "num_wedges": wedges,
            "global_clustering": c_global,
            "mean_local_clustering": c_local,
            "num_components": n_components,
        }

    return out


def _summary_table(
    e8: dict[str, dict[str, Any]],
    e8xe8: dict[str, dict[str, Any]],
) -> None:
    print()
    print("=" * 64)
    print("SUMMARY (clustering coefficients)")
    print("=" * 64)
    header = f"{'':<28s}{'Single E8':>16s}{'E8 ⊕ E8':>20s}"
    print(header)
    for name, (descr, _) in CONVENTIONS.items():
        a = e8[name]["global_clustering"]
        b = e8xe8[name]["global_clustering"]
        print(
            f"  Convention {name} ({descr:<14s}) "
            f"{a:>16.12f}{b:>20.12f}"
        )

    print()
    ref = E8_CLUSTERING_LITERATURE_CLAIM
    print(f"Reference claim (literature): 25/32 = {ref:.12f}")
    matches: list[str] = []
    for name in CONVENTIONS:
        a = e8[name]["global_clustering"]
        b = e8xe8[name]["global_clustering"]
        if abs(a - ref) < 1e-6:
            matches.append(f"E8/{name}")
        if abs(b - ref) < 1e-6:
            matches.append(f"E8xE8/{name}")
    print("Conventions matching reference within 1e-6: "
          + (", ".join(matches) if matches else "(none)"))


def _ip_distribution_table(roots: np.ndarray, label: str) -> dict[float, int]:
    g = roots @ roots.T
    n = roots.shape[0]
    iu = np.triu_indices(n, k=1)
    rounded = np.round(g[iu], 6)
    unique, counts = np.unique(rounded, return_counts=True)
    print(f"\nInner-product distribution ({label}, "
          f"{n*(n-1)//2:,} unordered pairs):")
    for v, c in zip(unique.tolist(), counts.tolist()):
        print(f"  ⟨α,β⟩ = {v:>+5.1f}: {c:>9,d}")
    return {float(v): int(c) for v, c in zip(unique, counts)}


def _module_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for mod in ("numpy", "networkx"):
        try:
            m = __import__(mod)
            versions[mod] = m.__version__
        except Exception:  # pragma: no cover
            versions[mod] = "unavailable"
    return versions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute the E8 ⊕ E8 root-graph clustering coefficient",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="enable DEBUG-level logging"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="WARNING-level logging only"
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.getcwd(), "results"),
        help="directory in which to write the JSON output",
    )
    args = parser.parse_args(argv)

    if args.verbose and args.quiet:
        parser.error("--verbose and --quiet are mutually exclusive")
    level = (
        logging.DEBUG if args.verbose
        else logging.WARNING if args.quiet
        else logging.INFO
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 64)
    print("E8 ⊕ E8 Root System — Clustering Coefficient Analysis")
    print("=" * 64)

    e8_roots = construct_e8_roots()
    e8xe8_roots = construct_e8xe8_roots()

    e8_norms_sq = np.einsum("ij,ij->i", e8_roots, e8_roots)
    print(f"Construction:")
    print(f"  Single E8: {e8_roots.shape[0]} roots in R^{e8_roots.shape[1]}, "
          f"squared norm = {e8_norms_sq.mean():.6f}")
    print(f"  E8 ⊕ E8:   {e8xe8_roots.shape[0]} roots in "
          f"R^{e8xe8_roots.shape[1]} (240 ⊕ 240, orthogonal blocks)")

    ip_e8 = _ip_distribution_table(e8_roots, "single E8")
    ip_e8xe8 = _ip_distribution_table(e8xe8_roots, "E8 ⊕ E8")

    manifest: dict[str, str] = {}
    e8_results = _analyze(e8_roots, "Single E8", manifest)
    e8xe8_results = _analyze(e8xe8_roots, "E8 ⊕ E8", manifest)

    _summary_table(e8_results, e8xe8_results)

    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f"clustering_{timestamp}.json")
    payload: dict[str, Any] = {
        "timestamp_utc": timestamp,
        "tolerance": INNER_PRODUCT_TOLERANCE,
        "module_versions": _module_versions(),
        "inner_product_distribution": {
            "e8": ip_e8,
            "e8xe8": ip_e8xe8,
        },
        "results": {
            "e8": e8_results,
            "e8xe8": e8xe8_results,
        },
        "literature_claim": E8_CLUSTERING_LITERATURE_CLAIM,
        "manifest_sha256": manifest,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"\nResults written to: {out_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
