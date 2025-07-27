"""
Construct and analyze the 2D toric code with two qubits per vertex.

This script builds the X‑ and Z‑type stabilizer check matrices for a two‑
dimensional periodic lattice of size ``L_x × L_y``.  Each vertex carries
two qubits (types ``A`` and ``B``).  Stabilizers are defined by the rules:

* **Z‑type stabilizer** at vertex ``(a,b)``:

  ::

      Z_A(a,b) · Z_A(a,b−1) · Z_B(a,b) · Z_B(a−1,b)

* **X‑type stabilizer** at vertex ``(a,b)``:

  ::

      X_A(a,b) · X_A(a+1,b) · X_B(a,b) · X_B(a,b+1)

These definitions correspond to those in the problem description, ensuring
that X and Z stabilizers commute locally.

For each pair of lattice dimensions ``L_x, L_y`` with 2 ≤ ``L_x``, ``L_y`` ≤ 5
the script constructs the stabilizer matrices, invokes the generic CSS
parameter calculator from ``CSS_code_parameters.py``, and prints the code
parameters including the minimal distances ``d_X`` and ``d_Z``.
"""

from __future__ import annotations

import itertools
from typing import List, Tuple

from CSS_code_parameters import compute_css_code_parameters


def qubit_index(a: int, b: int, qtype: str, Lx: int, Ly: int) -> int:
    """Return a unique index for the qubit of type ``qtype`` at (a,b)."""
    n_vertices = Lx * Ly
    vertex_id = (b % Ly) * Lx + (a % Lx)
    if qtype == 'A':
        return vertex_id
    elif qtype == 'B':
        return n_vertices + vertex_id
    else:
        raise ValueError(f"Unknown qubit type: {qtype}")


def build_2d_stabilizers(Lx: int, Ly: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Construct X and Z check matrices for the 2D toric code of given dimensions."""
    n_vertices = Lx * Ly
    n_qubits = 2 * n_vertices
    H_X: List[List[int]] = []
    H_Z: List[List[int]] = []
    zero_row = [0] * n_qubits
    for a in range(Lx):
        for b in range(Ly):
            # Z stabilizer: Z_A(a,b), Z_A(a,b-1), Z_B(a,b), Z_B(a-1,b)
            z_row = zero_row.copy()
            for da, db, qtype in [
                (0, 0, 'A'),
                (0, -1, 'A'),
                (0, 0, 'B'),
                (-1, 0, 'B'),
            ]:
                idx = qubit_index(a + da, b + db, qtype, Lx, Ly)
                z_row[idx] ^= 1
            H_Z.append(z_row)
            # X stabilizer: X_A(a,b), X_A(a+1,b), X_B(a,b), X_B(a,b+1)
            x_row = zero_row.copy()
            for da, db, qtype in [
                (0, 0, 'A'),
                (1, 0, 'A'),
                (0, 0, 'B'),
                (0, 1, 'B'),
            ]:
                idx = qubit_index(a + da, b + db, qtype, Lx, Ly)
                x_row[idx] ^= 1
            H_X.append(x_row)
    return H_X, H_Z


def main():
    print("Lx Ly | n r_X r_Z k d_X d_Z")
    for Lx, Ly in itertools.product(range(2, 6), repeat=2):
        H_X, H_Z = build_2d_stabilizers(Lx, Ly)
        params = compute_css_code_parameters(H_X, H_Z, calc_d_X=True, calc_d_Z=True)
        # Format distance values as strings to avoid formatting errors when values are ints
        dX = params['d_X'] if params['d_X'] is not None else '-'
        dZ = params['d_Z'] if params['d_Z'] is not None else '-'
        print(
            f"{Lx:2d} {Ly:2d} | {params['n']:3d} "
            f"{params['r_X']:3d} {params['r_Z']:3d} {params['k']:2d} "
            f"{str(dX):>2} "
            f"{str(dZ):>2}"
        )


if __name__ == "__main__":
    main()