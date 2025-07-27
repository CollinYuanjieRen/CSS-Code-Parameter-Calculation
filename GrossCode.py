"""
Generate and analyze the [[144,12,12]] Bivariate Bicycle (gross) code.

This script constructs the check matrices ``H_X`` and ``H_Z`` for the
bivariate bicycle code QC(A, B) described in Ref. [Bravyi et al., 2024]
【19139348279059†L624-L773】.  The construction is parameterized by two
integers ``ℓ`` and ``m`` determining the size of the underlying cyclic
lattice and two polynomials ``A`` and ``B`` in non‑commuting variables
``x`` and ``y``.  Specifically, one defines the shift operators

::

    x = S_ℓ ⊗ I_m,
    y = I_ℓ ⊗ S_m,

where ``S_k`` denotes the cyclic shift matrix of size ``k×k`` and ``I_k`` is the
identity.  The matrices ``A`` and ``B`` are sums of three monomials in
``x`` and ``y``, each monomial being a power of ``x`` or ``y``.  The
bivariate bicycle code QC(A, B) has check matrices

::

    H_X = [A | B],
    H_Z = [B^T | A^T],

each of size ``(ℓ m) × (2 ℓ m)``.  All arithmetic is performed over GF(2).

For the [[144,12,12]] code one chooses ℓ=12, m=6, and polynomials

    A = x^3 + y^1 + y^2,
    B = y^3 + x^1 + x^2.

The number of physical qubits is ``n = 2 ℓ m = 144`` and the number of
logical qubits is ``k = 2 ⋅ dim (ker A ∩ ker B) = 12``【19139348279059†L690-L694】.
The distance is ``d = 12``【19139348279059†L744-L748】.  This script
constructs the stabilizer matrices explicitly and verifies these parameters
using the generic CSS code parameter calculator.
"""

from __future__ import annotations

from typing import List, Tuple

from CSS_code_parameters import compute_css_code_parameters


def build_bb_code(
    ell: int,
    m: int,
    A_exponents: List[Tuple[str, int]],
    B_exponents: List[Tuple[str, int]],
) -> Tuple[List[List[int]], List[List[int]]]:
    """Construct H_X and H_Z for a bivariate bicycle code.

    Args:
        ell: Size of the first cyclic dimension (ℓ).
        m: Size of the second cyclic dimension (m).
        A_exponents: List of monomials in A, each given as a tuple (orient, exponent)
            where orient ∈ {'x','y'} and exponent is a non‑negative integer.
        B_exponents: Same as A_exponents but for B.

    Returns:
        (H_X, H_Z) as lists of lists of 0/1 integers.  ``H_X`` and ``H_Z``
        have shape (ℓ·m, 2·ℓ·m).
    """
    n0 = ell * m
    n_qubits = 2 * n0
    H_X: List[List[int]] = []
    H_Z: List[List[int]] = []
    # Precompute mod arithmetic for efficiency
    for r in range(n0):
        row_X = [0] * n_qubits
        # Determine lattice coordinates (i,j) for this row
        i = r // m
        j = r % m
        # Populate A part (first half of H_X row)
        for orient, exp in A_exponents:
            if orient == 'x':
                i2 = (i + exp) % ell
                j2 = j
            elif orient == 'y':
                i2 = i
                j2 = (j + exp) % m
            else:
                raise ValueError(f"Unknown orient {orient}")
            c = i2 * m + j2
            row_X[c] ^= 1
        # Populate B part (second half)
        for orient, exp in B_exponents:
            if orient == 'x':
                i2 = (i + exp) % ell
                j2 = j
            elif orient == 'y':
                i2 = i
                j2 = (j + exp) % m
            c = i2 * m + j2
            row_X[n0 + c] ^= 1
        H_X.append(row_X)
    # Construct H_Z as [B^T | A^T]
    for r in range(n0):
        row_Z = [0] * n_qubits
        i = r // m
        j = r % m
        # B^T part: we need rows r0_B where B maps to r
        for orient, exp in B_exponents:
            if orient == 'x':
                # (i0 + exp, j0) == (i,j) => i0 = i - exp, j0 = j
                i0 = (i - exp) % ell
                j0 = j
            elif orient == 'y':
                # (i0, j0 + exp) == (i,j) => i0 = i, j0 = j - exp
                i0 = i
                j0 = (j - exp) % m
            else:
                raise ValueError(f"Unknown orient {orient}")
            r0 = i0 * m + j0
            row_Z[r0] ^= 1
        # A^T part: rows r0_A where A maps to r
        for orient, exp in A_exponents:
            if orient == 'x':
                i0 = (i - exp) % ell
                j0 = j
            elif orient == 'y':
                i0 = i
                j0 = (j - exp) % m
            r0 = i0 * m + j0
            row_Z[n0 + r0] ^= 1
        H_Z.append(row_Z)
    return H_X, H_Z


def main():
    # Parameters for the [[144,12,12]] gross code
    ell = 12
    m = 6
    A_exponents = [('x', 3), ('y', 1), ('y', 2)]
    B_exponents = [('y', 3), ('x', 1), ('x', 2)]
    H_X, H_Z = build_bb_code(ell, m, A_exponents, B_exponents)
    params = compute_css_code_parameters(H_X, H_Z, calc_d_X=True, calc_d_Z=True)
    print("ell m | n r_X r_Z k d_X d_Z")
    # Format distance values as strings to avoid formatting errors when values are ints
    dX = params['d_X'] if params['d_X'] is not None else '-'
    dZ = params['d_Z'] if params['d_Z'] is not None else '-'
    print(
        f"{ell:3d} {m:3d} | {params['n']:4d} "
        f"{params['r_X']:3d} {params['r_Z']:3d} {params['k']:3d} "
        f"{str(dX):>3} "
        f"{str(dZ):>3}"
    )


if __name__ == "__main__":
    main()