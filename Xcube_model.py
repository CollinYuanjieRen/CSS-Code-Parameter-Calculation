"""
Construct and analyze the X‑cube fracton model on a three‑dimensional torus.

This script builds the X‑ and Z‑type stabilizer check matrices for the
3D X‑cube model on an ``L_x × L_y × L_z`` cubic lattice with periodic
boundary conditions.  Qubits live on edges of the lattice.  There are three
types of edges:

* **x‑edges** at integer coordinates ``(a,b,c)`` connecting ``(a,b,c)`` to
  ``(a+1,b,c)``.
* **y‑edges** at integer coordinates ``(a,b,c)`` connecting ``(a,b,c)`` to
  ``(a,b+1,c)``.
* **z‑edges** at integer coordinates ``(a,b,c)`` connecting ``(a,b,c)`` to
  ``(a,b,c+1)``.

The stabilizer generators are as follows:

* **Z‑type stabilizers**:  At each vertex ``(a,b,c)`` there are three
  stabilizers, one associated with each coordinate plane.  For planes
  ``xy`` (perpendicular to z), ``yz`` (perpendicular to x), and ``zx``
  (perpendicular to y), the stabilizer acts on four edges oriented along the
  two axes spanning the plane.  Explicitly, for the plane spanned by axes
  ``μ`` and ``ν``, the stabilizer on vertex ``(a,b,c)`` acts on the ``μ``‑edge
  at ``(a,b,c)`` and at ``(a,b,c)`` shifted by ``−ν``, and on the ``ν``‑edge
  at ``(a,b,c)`` and at ``(a,b,c)`` shifted by ``−μ``.  The shifts are
  interpreted modulo the lattice dimensions.

* **X‑type stabilizers**:  At each cube ``(a,b,c)`` there is a stabilizer
  acting on the twelve edges around the cube.  It includes the four x‑edges
  on the bottom and top faces, the four y‑edges on the front and back faces,
  and the four z‑edges on the left and right faces.  In terms of the
  canonical lower endpoint labeling of edges, these edges are:

  * x‑edges at ``(a,b,c)``, ``(a,b,c+1)``, ``(a,b+1,c)``, ``(a,b+1,c+1)``;
  * y‑edges at ``(a,b,c)``, ``(a+1,b,c)``, ``(a,b,c+1)``, ``(a+1,b,c+1)``;
  * z‑edges at ``(a,b,c)``, ``(a+1,b,c)``, ``(a,b+1,c)``, ``(a+1,b+1,c)``.

For each triple of lattice dimensions ``L_x, L_y, L_z`` with 2 ≤ L_x, L_y, L_z ≤ 4
this script constructs the stabilizer matrices and calls the generic CSS
parameter calculator to compute the code parameters, including minimal
distances.  Because the X‑cube model exhibits a subextensive ground state
degeneracy, the number of logical qubits is ``2(L_x + L_y + L_z) - 3``
【288260746202933†L1295-L1304】.  The code distance grows with system size; the
algorithm implemented here computes the exact minimal weights for small
lattices by exhaustive enumeration over the kernel basis.
"""

from __future__ import annotations

import itertools
from typing import List, Tuple

from CSS_code_parameters import compute_css_code_parameters


def edge_index(a: int, b: int, c: int, orient: str, Lx: int, Ly: int, Lz: int) -> int:
    """Return a unique index for the edge qubit at (a,b,c) with given orientation.

    Edge orientations are 'x', 'y', or 'z'.  The indexing is arranged so that
    all x‑edges come first, then y‑edges, then z‑edges.  Each orientation
    section contains ``L_x * L_y * L_z`` qubits.
    """
    a = a % Lx
    b = b % Ly
    c = c % Lz
    offset = (c * Ly + b) * Lx + a
    n_edges = Lx * Ly * Lz
    if orient == 'x':
        return offset
    elif orient == 'y':
        return n_edges + offset
    elif orient == 'z':
        return 2 * n_edges + offset
    else:
        raise ValueError(f"Unknown orientation: {orient}")


def build_xcube_stabilizers(Lx: int, Ly: int, Lz: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Construct X and Z check matrices for the X‑cube model on a torus.

    The lattice has periodic boundary conditions in all three directions.  Qubits live on edges oriented
    along the x, y, or z axes.  For each vertex there are three Z‑type stabilizers (one for each pair of axes)
    acting on four edges in the corresponding coordinate plane.  For each cube there is an X‑type stabilizer
    acting on the twelve edges around the cube.
    """
    n_edges = Lx * Ly * Lz
    n_qubits = 3 * n_edges
    H_X: List[List[int]] = []
    H_Z: List[List[int]] = []
    zero_row = [0] * n_qubits
    # Define shifts for each axis: used to shift coordinates by −1 along that axis
    shift = {
        'x': (-1, 0, 0),
        'y': (0, -1, 0),
        'z': (0, 0, -1),
    }
    # Z‑type stabilizers on vertices
    for a in range(Lx):
        for b in range(Ly):
            for c in range(Lz):
                # Each plane normal to a coordinate axis is spanned by two axes (μ, ν).
                # For the plane spanned by μ and ν, the cross operator (normal to the third axis)
                # acts on four edges: two oriented along μ and two oriented along ν.  The two μ‑edges
                # adjacent to (a,b,c) are the μ‑edge starting at (a,b,c) and the μ‑edge ending at (a,b,c),
                # which starts at (a,b,c) shifted by −μ.  Similarly for the ν‑edges.  We implement this
                # by shifting coordinates by −μ for μ‑edges and by −ν for ν‑edges.
                for mu, nu in [('x', 'y'), ('y', 'z'), ('z', 'x')]:
                    z_row = zero_row.copy()
                    # μ‑edges: at (a,b,c) and at (a,b,c) shifted by −μ
                    for delta in [(0, 0, 0), shift[mu]]:
                        aa = a + delta[0]
                        bb = b + delta[1]
                        cc = c + delta[2]
                        idx = edge_index(aa, bb, cc, mu, Lx, Ly, Lz)
                        z_row[idx] ^= 1
                    # ν‑edges: at (a,b,c) and at (a,b,c) shifted by −ν
                    for delta in [(0, 0, 0), shift[nu]]:
                        aa = a + delta[0]
                        bb = b + delta[1]
                        cc = c + delta[2]
                        idx = edge_index(aa, bb, cc, nu, Lx, Ly, Lz)
                        z_row[idx] ^= 1
                    H_Z.append(z_row)
    # X‑type stabilizers on cubes
    for a in range(Lx):
        for b in range(Ly):
            for c in range(Lz):
                x_row = zero_row.copy()
                # x‑edges on bottom and top faces
                for da, db, dc in [
                    (0, 0, 0),
                    (0, 0, 1),
                    (0, 1, 0),
                    (0, 1, 1),
                ]:
                    idx = edge_index(a + da, b + db, c + dc, 'x', Lx, Ly, Lz)
                    x_row[idx] ^= 1
                # y‑edges on left and right faces
                for da, db, dc in [
                    (0, 0, 0),
                    (1, 0, 0),
                    (0, 0, 1),
                    (1, 0, 1),
                ]:
                    idx = edge_index(a + da, b + db, c + dc, 'y', Lx, Ly, Lz)
                    x_row[idx] ^= 1
                # z‑edges on front and back faces
                for da, db, dc in [
                    (0, 0, 0),
                    (1, 0, 0),
                    (0, 1, 0),
                    (1, 1, 0),
                ]:
                    idx = edge_index(a + da, b + db, c + dc, 'z', Lx, Ly, Lz)
                    x_row[idx] ^= 1
                H_X.append(x_row)
    return H_X, H_Z


def main():
    print("Lx Ly Lz | n r_X r_Z k d_X d_Z")
    for Lx, Ly, Lz in itertools.product(range(2, 5), repeat=3):
        # Build stabilizers for this lattice size
        H_X, H_Z = build_xcube_stabilizers(Lx, Ly, Lz)
        # If Lx==2 and Ly,Lz <=3, print the support of each stabilizer for debugging
        if Lx == 2 and Ly <= 3 and Lz <= 3:
            print(f"\nSupports for lattice size ({Lx},{Ly},{Lz}):")
            # Print Z stabilizers
            print("Z stabilizers:")
            shift = {'x': (-1,0,0), 'y': (0,-1,0), 'z': (0,0,-1)}
            planes = [('x','y'),('y','z'),('z','x')]
            for a in range(Lx):
                for b in range(Ly):
                    for c in range(Lz):
                        for mu, nu in planes:
                            edges = []
                            # μ edges: at (a,b,c) and at (a,b,c) shifted by −μ
                            for delta in [(0,0,0), shift[mu]]:
                                aa = a + delta[0]; bb = b + delta[1]; cc = c + delta[2]
                                edges.append((mu, ((aa)%Lx, (bb)%Ly, (cc)%Lz)))
                            # ν edges: at (a,b,c) and at (a,b,c) shifted by −ν
                            for delta in [(0,0,0), shift[nu]]:
                                aa = a + delta[0]; bb = b + delta[1]; cc = c + delta[2]
                                edges.append((nu, ((aa)%Lx, (bb)%Ly, (cc)%Lz)))
                            print(f"  Z({a},{b},{c}) plane {mu}{nu}: {edges}")
            # Print X stabilizers
            print("X stabilizers:")
            for a in range(Lx):
                for b in range(Ly):
                    for c in range(Lz):
                        edges = []
                        # x‑edges
                        for da,db,dc in [(0,0,0),(0,0,1),(0,1,0),(0,1,1)]:
                            edges.append(('x', (((a+da)%Lx), ((b+db)%Ly), ((c+dc)%Lz))))
                        # y‑edges
                        for da,db,dc in [(0,0,0),(1,0,0),(0,0,1),(1,0,1)]:
                            edges.append(('y', (((a+da)%Lx), ((b+db)%Ly), ((c+dc)%Lz))))
                        # z‑edges
                        for da,db,dc in [(0,0,0),(1,0,0),(0,1,0),(1,1,0)]:
                            edges.append(('z', (((a+da)%Lx), ((b+db)%Ly), ((c+dc)%Lz))))
                        print(f"  X cube at ({a},{b},{c}): {edges}")
        # Compute code parameters and distances
        params = compute_css_code_parameters(H_X, H_Z, calc_d_X=True, calc_d_Z=True)
        dX = params['d_X'] if params['d_X'] is not None else '-'
        dZ = params['d_Z'] if params['d_Z'] is not None else '-'
        print(
            f"{Lx:2d} {Ly:2d} {Lz:2d} | {params['n']:4d} "
            f"{params['r_X']:3d} {params['r_Z']:3d} {params['k']:2d} "
            f"{str(dX):>3} "
            f"{str(dZ):>3}"
        )


if __name__ == "__main__":
    main()