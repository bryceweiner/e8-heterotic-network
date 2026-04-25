"""
Construction of the E8 and E8 ⊕ E8 root systems from first principles.

No scaling factors. No engineered cross-couplings. No deformations.
Every E8 root has squared norm exactly 2; every cross-block inner product
in E8 ⊕ E8 is exactly 0.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def construct_e8_roots() -> np.ndarray:
    """Return the 240 roots of E8 as a (240, 8) float64 array.

    The E8 root system splits into:

    * 112 "integer" / D8 roots: ``±e_i ± e_j`` for ``1 ≤ i < j ≤ 8`` with
      both signs chosen independently. Count: ``4 * C(8, 2) = 112``.
    * 128 "half-integer" / spinor roots: ``(±1/2, ±1/2, ..., ±1/2)`` with an
      even number of minus signs. Count: ``2^7 = 128``.

    Every root has squared norm exactly 2. Distinct non-antipodal roots have
    inner products in ``{-1, 0, +1}``; antipodal pairs give ``-2``.
    """
    roots: list[np.ndarray] = []

    # Integer roots: ±e_i ± e_j, i < j.
    for i in range(8):
        for j in range(i + 1, 8):
            for sign_i in (1.0, -1.0):
                for sign_j in (1.0, -1.0):
                    r = np.zeros(8, dtype=np.float64)
                    r[i] = sign_i
                    r[j] = sign_j
                    roots.append(r)

    # Half-integer roots: (±1/2)^8 with an even number of minus signs.
    for mask in range(256):
        signs = np.empty(8, dtype=np.float64)
        minus_count = 0
        for k in range(8):
            if (mask >> k) & 1:
                signs[k] = -0.5
                minus_count += 1
            else:
                signs[k] = 0.5
        if minus_count % 2 == 0:
            roots.append(signs)

    arr = np.asarray(roots, dtype=np.float64)
    if arr.shape != (240, 8):
        raise ValueError(
            f"Expected E8 to have shape (240, 8); got {arr.shape}"
        )

    logger.debug("Constructed E8 root system: shape=%s", arr.shape)
    return arr


def construct_e8xe8_roots() -> np.ndarray:
    """Return the 480 roots of E8 ⊕ E8 as a (480, 16) float64 array.

    Layout:

    * Rows 0..239: the 240 E8 roots in coordinates 0..7, zeros in 8..15.
    * Rows 240..479: the 240 E8 roots in coordinates 8..15, zeros in 0..7.

    Every root has squared norm 2. Inner products between rows in different
    blocks are exactly 0.
    """
    e8 = construct_e8_roots()

    arr = np.zeros((480, 16), dtype=np.float64)
    arr[:240, :8] = e8
    arr[240:, 8:] = e8

    logger.debug("Constructed E8 ⊕ E8 root system: shape=%s", arr.shape)
    return arr


def construct_cartan_subalgebra() -> np.ndarray:
    """Return 8 simple roots of E8 forming a basis of the Cartan subalgebra.

    Uses the standard Bourbaki choice for E8:

        α₁ = (1/2)(e₁ - e₂ - e₃ - e₄ - e₅ - e₆ - e₇ + e₈)
        α₂ = e₁ + e₂
        α₃ = e₂ - e₁
        α₄ = e₃ - e₂
        α₅ = e₄ - e₃
        α₆ = e₅ - e₄
        α₇ = e₆ - e₅
        α₈ = e₇ - e₆

    These are 8 of the 240 E8 roots. They are NOT additional roots — they
    are provided as a basis for callers that need one. The clustering
    computation does not use this function.
    """
    a = np.zeros((8, 8), dtype=np.float64)
    a[0] = np.array(
        [0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5], dtype=np.float64
    )
    a[1] = np.array([1.0, 1.0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    a[2] = np.array([-1.0, 1.0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    a[3] = np.array([0, -1.0, 1.0, 0, 0, 0, 0, 0], dtype=np.float64)
    a[4] = np.array([0, 0, -1.0, 1.0, 0, 0, 0, 0], dtype=np.float64)
    a[5] = np.array([0, 0, 0, -1.0, 1.0, 0, 0, 0], dtype=np.float64)
    a[6] = np.array([0, 0, 0, 0, -1.0, 1.0, 0, 0], dtype=np.float64)
    a[7] = np.array([0, 0, 0, 0, 0, -1.0, 1.0, 0], dtype=np.float64)
    return a
