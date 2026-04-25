# E8 ⊕ E8 Root System: Honest Clustering Analysis

A Python library that constructs the E8 and E8 ⊕ E8 root systems from
first principles and reports their adjacency-graph clustering coefficients
under four canonical conventions.

## What this library does

1. Constructs the standard E8 root system (240 roots in ℝ⁸, all squared
   norm 2) from explicit formulas.
2. Constructs E8 ⊕ E8 as the direct sum (480 roots in ℝ¹⁶, organised as
   two orthogonal blocks of 240 roots each).
3. Defines four adjacency conventions, each by a single rule on the inner
   product matrix.
4. Counts triangles (via `trace(A³)/6`, cross-checked against NetworkX)
   and wedges, and reports global / mean-local clustering for each
   convention on each system.

## Headline numbers

Both single E8 and E8 ⊕ E8 produce identical clustering values, since
the orthogonal blocks split the direct-sum graph into two disjoint
copies of the E8 root graph.

| Convention | Edge rule       | Degree | Triangles | C_global       | Closed form |
|:----------:|-----------------|:------:|----------:|---------------:|:-----------:|
| A          | ⟨α, β⟩ = +1     | 56     | 60,480    | 0.4909090909   | 27/55       |
| B          | ⟨α, β⟩ ≠ 0      | 113    | 264,320   | 0.5221238938   | 59/113      |
| C          | \|⟨α, β⟩\| = 1  | 112    | 250,880   | 0.5045045045   | 56/111      |
| D          | ⟨α, β⟩ = −1     | 56     | 2,240     | 0.0181818182   | 1/55        |


## Installation

```bash
pip install -e .
```

The core analysis only requires `numpy` and `networkx`. The optional
PyTorch layer requires `torch` in addition.

## Running the analysis

```bash
python -m e8_heterotic.cli.compute_clustering
```

This prints the inner-product distribution, per-convention degree /
edge / triangle / wedge / clustering counts for both single E8 and
E8 ⊕ E8, and writes a JSON results file (with a SHA-256
reproducibility manifest) to `results/clustering_<timestamp>.json`.
Two runs on the same library versions produce byte-identical manifest
hashes.

Flags:

- `--verbose` — DEBUG-level logging
- `--quiet`   — WARNING-level logging only
- `--results-dir <path>` — override the JSON output directory

## Library API

```python
from e8_heterotic import (
    construct_e8_roots,
    construct_e8xe8_roots,
    adjacency_inner_product_one,         # Convention A
    adjacency_inner_product_nonzero,     # Convention B
    adjacency_absolute_inner_product_one,  # Convention C
    adjacency_inner_product_minus_one,   # Convention D
    count_triangles_and_wedges,
    global_clustering_coefficient,
    mean_local_clustering_coefficient,
    E8xE8RootSystem,
)

roots = construct_e8xe8_roots()                 # (480, 16)
adjacency = adjacency_inner_product_one(roots)  # (480, 480) bool
triangles, wedges = count_triangles_and_wedges(adjacency)
c_global = global_clustering_coefficient(adjacency)

# Or use the orchestrator for all four conventions at once:
system = E8xE8RootSystem()
results = system.analyze('e8xe8')
```

## PyTorch layer

The PyTorch layer at `e8_heterotic.core.network.E8E8Layer` uses the
honest 480-vertex E8 ⊕ E8 root graph. The clustering coefficient it
reports is whatever the chosen adjacency convention produces — there is
no longer an assertion that the value equals any pre-determined target.

```python
from e8_heterotic import E8E8Layer

layer = E8E8Layer(input_dim=128, output_dim=64, adjacency_convention='A')
print(f"Computed clustering: {layer.get_clustering_coefficient():.10f}")
# 0.4909090909
```

## Module layout

```
e8_heterotic/
├── core/
│   ├── constants.py        # structural constants and tolerances
│   ├── root_system.py      # construct_e8_roots, construct_e8xe8_roots
│   ├── adjacency.py        # the four adjacency conventions
│   ├── clustering.py       # triangle/wedge counting, clustering
│   ├── construction.py     # E8xE8RootSystem orchestrator
│   ├── cache.py            # disk cache for root system + adjacency
│   └── network.py          # PyTorch layer (optional)
├── cli/
│   └── compute_clustering.py   # CLI driver
├── utils/
│   ├── mathematics.py      # geometric helpers (deprecated stubs)
│   └── device.py           # PyTorch device selection
└── tests/
    ├── test_root_system.py # 240/480 roots, norms, IP distribution
    └── test_clustering.py  # triangle counting, regularity, hardcode guard
```

## Running the tests

```bash
python -m pytest e8_heterotic/tests
```

The suite checks:

- Root counts (240, 480), squared norms (=2), inner-product distribution,
  closure under negation.
- Triangle counts agree between `trace(A³)/6` and NetworkX on small
  graphs and on the full E8 graph.
- E8 graph regularity for each convention (degrees 56, 113, 112, 56).
- Conventions A and D do *not* produce identical clustering values
  (they share edge count but have very different triangle counts).
- Empty / path / Petersen graphs return clustering 0, ruling out
  hardcoded returns.

## License

MIT — see [LICENSE](LICENSE).
