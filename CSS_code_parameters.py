"""
CSS code parameter calculator.

This module defines a function ``compute_css_code_parameters`` that accepts the
binary check matrices ``H_X`` and ``H_Z`` of a Calderbank–Shor–Steane (CSS)
stabilizer code and returns the code parameters.  The matrices must be
provided as lists of lists of integers 0/1, where each inner list represents
one stabilizer generator and has length equal to the number of physical
qubits ``n``.  All arithmetic is performed over the finite field GF(2).

The returned dictionary contains the following keys:

``n``
    The number of physical qubits.

``r_X``
    The rank (over GF(2)) of ``H_X``.

``r_Z``
    The rank (over GF(2)) of ``H_Z``.

``k``
    The number of logical qubits, computed as ``n - r_X - r_Z``.

``d_X``
    The minimum weight of a non‑trivial logical X operator.  If the boolean
    ``calc_d_X`` is passed as ``False`` then ``d_X`` will be ``None``.

``d_Z``
    The minimum weight of a non‑trivial logical Z operator.  If the boolean
    ``calc_d_Z`` is passed as ``False`` then ``d_Z`` will be ``None``.

To compute code distances the module enumerates all non‑zero combinations of
kernel basis vectors.  This is feasible only when the number of logical qubits
is modest (up to a few dozen) and therefore should be disabled for very large
codes.

The module also defines several internal helper functions for common linear
algebra tasks over GF(2), implemented using Python integers to represent
bitvectors for efficiency.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple, Any

# -----------------------------------------------------------------------------
# Logging helper
#
# Define a simple function ``log_print`` that writes messages both to standard
# output and to a persistent log file named ``temporary_log.txt`` in the
# current working directory.  This allows all modules using this file to
# record the same information to a log.  Note that the file is opened in
# append mode for each call to avoid truncating earlier entries.

def log_print(*objects: Any, sep: str = " ", end: str = "\n") -> None:
    """Print to stdout and append the same message to ``temporary_log.txt``.

    Args:
        *objects: Objects to print (like the built‑in ``print``).
        sep: Separator string to insert between objects.
        end: End‑of‑line string appended after the last object.
    """
    text = sep.join(str(o) for o in objects) + end
    try:
        with open("temporary_log.txt", "a", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        # If writing to the log fails for any reason, silently continue
        pass
    print(*objects, sep=sep, end=end)


def _bitlist_to_int(bits: Iterable[int]) -> int:
    """Convert an iterable of 0/1 integers into a Python integer bitmask.

    The least significant bit corresponds to index 0 in the iterable.
    """
    x = 0
    for i, b in enumerate(bits):
        if b & 1:
            x |= 1 << i
    return x


def _gf2_row_echelon(matrix_ints: List[int], width: int) -> Tuple[List[int], List[int]]:
    """Perform Gaussian elimination over GF(2) on the given matrix.

    Args:
        matrix_ints: A list of integers, each representing a row of the matrix as a bitmask.
        width: The number of columns of the matrix (i.e., the number of bits).

    Returns:
        A tuple (reduced_rows, pivot_cols) where ``reduced_rows`` is a list of row
        bitmasks in row-echelon form (containing the pivots) and ``pivot_cols``
        contains the column indices of pivot positions.

    The function destructively modifies ``matrix_ints``.  Rows are processed
    from the highest column index down to zero to align with typical GF(2)
    elimination procedures.
    """
    rows = matrix_ints.copy()
    pivot_cols: List[int] = []
    r = 0
    # Iterate over column indices from most significant down to zero
    for col in reversed(range(width)):
        # Find a row at or below r with a 1 in this column
        pivot_row = None
        for i in range(r, len(rows)):
            if (rows[i] >> col) & 1:
                pivot_row = i
                break
        if pivot_row is None:
            continue
        # Swap into position r
        rows[r], rows[pivot_row] = rows[pivot_row], rows[r]
        pivot_cols.append(col)
        # Eliminate this bit from all other rows
        for i in range(len(rows)):
            if i != r and ((rows[i] >> col) & 1):
                rows[i] ^= rows[r]
        r += 1
        if r == len(rows):
            break
    return rows[:r], pivot_cols


def _gf2_rank(matrix: List[List[int]]) -> int:
    """Compute the rank of a binary matrix over GF(2).

    Args:
        matrix: A list of lists of 0/1 values.

    Returns:
        The rank of the matrix.
    """
    if not matrix:
        return 0
    width = len(matrix[0])
    # Convert rows to integers for efficient arithmetic
    matrix_ints = [_bitlist_to_int(row) for row in matrix]
    reduced_rows, _ = _gf2_row_echelon(matrix_ints, width)
    return len(reduced_rows)


def _row_space_basis_int(matrix: List[List[int]]) -> List[int]:
    """Compute a basis for the row space of a binary matrix over GF(2).

    The basis is returned as a list of bitmasks representing rows.
    """
    if not matrix:
        return []
    width = len(matrix[0])
    matrix_ints = [_bitlist_to_int(row) for row in matrix]
    reduced_rows, _ = _gf2_row_echelon(matrix_ints, width)
    return reduced_rows


def _kernel_basis_int(matrix: List[List[int]]) -> List[int]:
    """Compute a basis for the kernel (null space) of a binary matrix over GF(2).

    Args:
        matrix: A list of lists of 0/1 values representing the matrix with shape m×n.

    Returns:
        A list of ``n``-bit integers forming a basis for the null space of the matrix.
    """
    if not matrix:
        return []  # The full space is the kernel; this case isn't encountered for codes
    m = len(matrix)
    n = len(matrix[0])
    # Convert rows to bitmasks
    matrix_ints = [_bitlist_to_int(row) for row in matrix]
    # Row‑reduce and track pivot columns
    reduced_rows, pivot_cols = _gf2_row_echelon(matrix_ints, n)
    pivot_set = set(pivot_cols)
    free_cols = [col for col in range(n) if col not in pivot_set]
    # Precompute mapping from pivot col to row index
    pivot_to_row = dict(zip(pivot_cols, range(len(pivot_cols))))
    basis: List[int] = []
    # For each free column, generate a null space basis vector
    for free_col in free_cols:
        # Start with vector having a 1 in the free column
        vec = 1 << free_col
        # For each pivot, if the pivot row has a 1 in free_col, include that pivot
        for pivot_col, row_idx in zip(pivot_cols, range(len(pivot_cols))):
            row = reduced_rows[row_idx]
            if (row >> free_col) & 1:
                vec |= 1 << pivot_col
        basis.append(vec)
    return basis


def _reduce_by_basis(vec: int, basis: List[int]) -> int:
    """Reduce a vector by a row space basis to determine membership in the span.

    Args:
        vec: A bitmask representing the vector to test.
        basis: A list of bitmasks forming a row space basis.

    Returns:
        The remainder of reducing ``vec`` by ``basis``.  If the remainder is zero,
        then ``vec`` lies in the span of ``basis``.
    """
    x = vec
    # Sort basis in descending order of least significant bit positions for stability
    sorted_basis = sorted(basis, reverse=True)
    for b in sorted_basis:
        # If x has a 1 in the pivot position of b, eliminate it
        # Find pivot bit of b: the highest bit set
        if x == 0:
            break
        pivot = b.bit_length() - 1
        if (x >> pivot) & 1:
            x ^= b
    return x


def _compute_min_logical_weight(kernel_basis: List[int], row_space_basis: List[int]) -> Optional[int]:
    """Compute the minimum weight of non‑trivial logical operators.

    A non‑trivial logical operator is an element of the kernel of one check
    matrix that is not contained in the row space of the other check matrix.
    The weight of an operator is the number of qubits on which it acts non‑
    trivially (i.e., the Hamming weight of the bitmask).

    Args:
        kernel_basis: A list of bitmasks forming a basis for the kernel of the
            check matrix defining the syndrome that anti‑commutes with the
            operator type of interest.
        row_space_basis: A list of bitmasks forming a basis for the row space
            of the other check matrix (operators that must be quotiented out).

    Returns:
        The minimum Hamming weight of any non‑zero vector in the span of
        ``kernel_basis`` that is not contained in the span of ``row_space_basis``.
        Returns ``None`` if no such vector exists (for instance, when there are
        no logical qubits).
    """
    # If there are no kernel vectors, there are no logical operators of this type.
    if not kernel_basis:
        return None
    # Reduce each kernel basis vector by the row space.  Any kernel vector
    # whose remainder is zero lies entirely in the row space and does not
    # contribute to the logical operator space.  Collect the non‑zero
    # remainders as potential generators for the logical space.
    coset_vectors = []
    for v in kernel_basis:
        rem = _reduce_by_basis(v, row_space_basis) if row_space_basis else v
        if rem != 0:
            coset_vectors.append(rem)
    # If all kernel vectors lie in the row space, there are no logical qubits.
    if not coset_vectors:
        return None
    # Perform GF(2) row reduction on the coset vectors to obtain an
    # independent basis for the logical operator space.  The width for row
    # reduction is given by the maximum bit length across vectors.
    width = max(vec.bit_length() for vec in coset_vectors)
    reduced_rows, _ = _gf2_row_echelon(coset_vectors, width)
    logical_basis = reduced_rows
    k = len(logical_basis)
    if k == 0:
        return None
    # Reduce each logical basis vector again to ensure it is not further
    # reducible by the row space.  This is typically redundant but provides
    # safety if the row reduction produced vectors that intersect the row space.
    if row_space_basis:
        logical_basis = [ _reduce_by_basis(v, row_space_basis) for v in logical_basis ]
    # Enumerate all non‑zero linear combinations of the logical basis.  The
    # number of logical qubits is ``k`` so there are ``2^k - 1`` such
    # combinations.  Compute the Hamming weight of each combination and keep
    # track of the minimal weight.  If weight one is found we can exit early.
    min_weight: Optional[int] = None
    for mask in range(1, 1 << k):
        vec = 0
        # Build the combination by XORing selected basis vectors
        for i in range(k):
            if (mask >> i) & 1:
                vec ^= logical_basis[i]
        # If the resulting vector lies in the row space, skip it (it is
        # equivalent to zero in the logical quotient space).  This should not
        # occur if ``logical_basis`` is properly reduced, but the check adds
        # robustness.
        if row_space_basis and _reduce_by_basis(vec, row_space_basis) == 0:
            continue
        w = vec.bit_count()
        if w == 0:
            continue
        if min_weight is None or w < min_weight:
            min_weight = w
            if min_weight == 1:
                return 1
    return min_weight


def _compute_min_logical_operator(kernel_basis: List[int], row_space_basis: List[int]) -> Tuple[Optional[int], Optional[int]]:
    """Find the minimum‑weight non‑trivial logical operator and return it.

    This function mirrors ``_compute_min_logical_weight`` but also returns a
    representative vector (bitmask) achieving the minimal Hamming weight.  A
    non‑trivial logical operator is an element of the kernel of one check
    matrix that is not contained in the row space of the other check matrix.

    Args:
        kernel_basis: A list of bitmasks forming a basis for the kernel of the
            check matrix defining the syndrome that anti‑commutes with the
            operator type of interest.
        row_space_basis: A list of bitmasks forming a basis for the row space
            of the other check matrix (operators that must be quotiented out).

    Returns:
        A tuple ``(min_weight, vector)`` where ``min_weight`` is the minimum
        Hamming weight of any non‑zero vector in the span of ``kernel_basis``
        that is not contained in the span of ``row_space_basis``, and
        ``vector`` is one bitmask representative achieving this weight.  If
        no such vector exists, returns ``(None, None)``.
    """
    # No kernel vectors means no logical operators of this type
    if not kernel_basis:
        return None, None
    # Reduce basis vectors by the row space to compute coset representatives
    coset_vectors = []
    for v in kernel_basis:
        rem = _reduce_by_basis(v, row_space_basis) if row_space_basis else v
        if rem != 0:
            coset_vectors.append(rem)
    if not coset_vectors:
        return None, None
    # Row reduce coset vectors to obtain independent logical basis
    width = max(vec.bit_length() for vec in coset_vectors)
    reduced_rows, _ = _gf2_row_echelon(coset_vectors, width)
    logical_basis = reduced_rows
    # Ensure logical basis vectors are not further reducible by the row space
    if row_space_basis:
        logical_basis = [ _reduce_by_basis(v, row_space_basis) for v in logical_basis ]
    k = len(logical_basis)
    if k == 0:
        return None, None
    # Enumerate all non‑zero combinations of logical basis vectors
    min_weight: Optional[int] = None
    min_vec: Optional[int] = None
    for mask in range(1, 1 << k):
        vec = 0
        for i in range(k):
            if (mask >> i) & 1:
                vec ^= logical_basis[i]
        # Skip if in row space
        if row_space_basis and _reduce_by_basis(vec, row_space_basis) == 0:
            continue
        # Skip zero vector
        if vec == 0:
            continue
        w = vec.bit_count()
        if min_weight is None or w < min_weight:
            min_weight = w
            min_vec = vec
            # Early exit if weight 1 found
            if min_weight == 1:
                break
    return min_weight, min_vec


def int_to_bitlist(x: int, length: int) -> List[int]:
    """Convert an integer bitmask back into a list of bits of the given length."""
    return [(x >> i) & 1 for i in range(length)]


def compute_css_code_parameters(
    H_X: List[List[int]],
    H_Z: List[List[int]],
    calc_d_X: bool = True,
    calc_d_Z: bool = True,
) -> dict:
    """Compute parameters of a CSS code defined by X and Z check matrices.

    Args:
        H_X: X‑type check matrix with shape m_X × n.
        H_Z: Z‑type check matrix with shape m_Z × n.
        calc_d_X: If True, compute the minimal weight of non‑trivial logical X operators.
        calc_d_Z: If True, compute the minimal weight of non‑trivial logical Z operators.

    Returns:
        A dictionary with keys ``n``, ``r_X``, ``r_Z``, ``k``, ``d_X`` and ``d_Z`` as
        described in the module docstring.
    """
    if not H_X and not H_Z:
        raise ValueError("At least one of H_X or H_Z must be non‑empty")
    # Determine the number of qubits from the first matrix provided
    n = len(H_X[0]) if H_X else len(H_Z[0])
    # Compute ranks
    r_X = _gf2_rank(H_X)
    r_Z = _gf2_rank(H_Z)
    # Compute number of logical qubits
    k = n - r_X - r_Z
    # Compute row space bases and kernel bases only if needed
    d_X = None
    d_Z = None
    min_vec_X: Optional[int] = None
    min_vec_Z: Optional[int] = None
    if calc_d_X and k > 0:
        # Kernel of H_Z, row space of H_X
        kernel_basis_Z = _kernel_basis_int(H_Z)
        row_basis_X = _row_space_basis_int(H_X)
        d_X, min_vec_X = _compute_min_logical_operator(kernel_basis_Z, row_basis_X)
    if calc_d_Z and k > 0:
        # Kernel of H_X, row space of H_Z
        kernel_basis_X = _kernel_basis_int(H_X)
        row_basis_Z = _row_space_basis_int(H_Z)
        d_Z, min_vec_Z = _compute_min_logical_operator(kernel_basis_X, row_basis_Z)
    # Prepare result dictionary
    result = {
        "n": n,
        "r_X": r_X,
        "r_Z": r_Z,
        "k": k,
        "d_X": d_X,
        "d_Z": d_Z,
    }
    # Include supports of minimal logical operators as lists of qubit indices
    if min_vec_X is not None:
        result["min_support_X"] = [i for i in range(n) if (min_vec_X >> i) & 1]
    else:
        result["min_support_X"] = None
    if min_vec_Z is not None:
        result["min_support_Z"] = [i for i in range(n) if (min_vec_Z >> i) & 1]
    else:
        result["min_support_Z"] = None
    # If either minimal vector exists, log its support to the log file and stdout
    if min_vec_X is not None:
        # Include the distance value in the message
        log_print(
            f"the X logical operator that has the smallest length d_X={d_X} is supported on qubits:",
            result["min_support_X"]
        )
    if min_vec_Z is not None:
        log_print(
            f"the Z logical operator that has the smallest length d_Z={d_Z} is supported on qubits:",
            result["min_support_Z"]
        )
    log_print(
        f"=============================="
    )
    return result
