# Author: Yuqi Zhang
"""MJ (Miyazawa-Jernigan) residue interaction matrix adapter.

Reads the project's existing MJ table at ``ras_folding/mj_matrix.txt``.

File format (as inspected on this project):
  - Line 1: header — 20 one-letter codes, whitespace-separated.
    Order in this file: C M F I L V W Y A G T S N Q D E H R K P.
  - Lines 2-21: 20 rows of 20 floats each, upper-triangular layout
    (row[i][j] is meaningful only when j >= i; cells with j < i are
    written as 0.00). Every meaningful cell is non-positive.

Sign convention
---------------
This file follows the standard MJ convention: negative = favorable
contact, positive = unfavorable. Inspecting the numerical content of
``ras_folding/mj_matrix.txt`` confirms this (hydrophobic-hydrophobic
diagonals such as C-C = -5.44, F-F = -7.26, L-L = -7.37 are deeply
negative; polar/charged diagonals such as K-K = -0.12 are weakly
negative; no positive entries exist). load_mj_table_default therefore
returns a table with sign_convention="negative_favorable".

For arbitrary user-supplied tables, sign_convention defaults to
"unknown" and the caller (FilterHamiltonian) is REQUIRED to skip
``H_bad_contact_proxy`` and surface ``mj_sign_unknown=True`` in summary.
This avoids using bad-contact filtering with an unverified table.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# Default location, relative to package root.
_DEFAULT_MJ_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mj_matrix.txt",
)


@dataclass
class MJContactTable:
    """Residue-pair interaction lookup, symmetric.

    Attributes
    ----------
    residues : list of one-letter AA codes in column/row order
    matrix : (R, R) float64, symmetric
    sign_convention : "negative_favorable" | "positive_favorable" | "unknown"
    source_path : str | None
    notes : dict — free-form provenance
    """
    residues: List[str]
    matrix: np.ndarray
    sign_convention: str = "unknown"
    source_path: Optional[str] = None
    notes: Dict[str, str] = field(default_factory=dict)
    _index: Dict[str, int] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.residues)
        if self.matrix.shape != (n, n):
            raise ValueError(
                f"matrix shape {self.matrix.shape} does not match "
                f"residues length {n}"
            )
        if not np.allclose(self.matrix, self.matrix.T, atol=1e-12):
            raise ValueError("matrix must be symmetric")
        if self.sign_convention not in (
            "negative_favorable", "positive_favorable", "unknown",
        ):
            raise ValueError(
                f"unsupported sign_convention {self.sign_convention!r}"
            )
        self._index = {aa.upper(): i for i, aa in enumerate(self.residues)}

    # -- queries ---------------------------------------------------------
    def has_residue(self, aa: str) -> bool:
        return aa.upper() in self._index

    def get_weight(
        self, residue_i: str, residue_j: str,
    ) -> Optional[float]:
        """Return signed w_ij or None if either residue is unknown.

        Alias of get_signed_weight, kept for backward compatibility.
        """
        return self.get_signed_weight(residue_i, residue_j)

    def get_signed_weight(
        self, residue_i: str, residue_j: str,
    ) -> Optional[float]:
        """Return the raw signed MJ weight (negative=favorable for the
        canonical project file). None if either residue is unknown."""
        i = self._index.get(residue_i.upper())
        j = self._index.get(residue_j.upper())
        if i is None or j is None:
            return None
        return float(self.matrix[i, j])

    def get_attraction_strength(
        self, residue_i: str, residue_j: str,
    ) -> Optional[float]:
        """Return non-negative attraction strength s_ij = max(-w_ij, 0).

        Only meaningful when sign_convention == "negative_favorable".
        Returns None if:
          - either residue is unknown, OR
          - sign_convention is unknown (so we cannot tell which sign
            means "favorable").
        For sign_convention == "positive_favorable", returns max(w_ij, 0).
        """
        if self.sign_convention == "unknown":
            return None
        w = self.get_signed_weight(residue_i, residue_j)
        if w is None:
            return None
        if self.sign_convention == "negative_favorable":
            return float(max(-w, 0.0))
        # positive_favorable
        return float(max(w, 0.0))

    def weight_matrix_for_sequence(
        self, sequence: Iterable[str],
    ) -> Tuple[np.ndarray, List[int]]:
        """Build an (L, L) per-position SIGNED weight matrix.

        Returns
        -------
        W : (L, L) float64; W[i, j] = self.matrix[ self._index[seq[i]],
            self._index[seq[j]] ] when both are known, else 0.0.
        missing : list of indices into `sequence` whose AA was not in
            this table.
        """
        seq = list(sequence)
        L = len(seq)
        W = np.zeros((L, L), dtype=np.float64)
        missing: List[int] = []
        idx_arr: List[Optional[int]] = []
        for k, aa in enumerate(seq):
            ix = self._index.get(str(aa).upper())
            idx_arr.append(ix)
            if ix is None:
                missing.append(k)
        for i in range(L):
            ii = idx_arr[i]
            if ii is None:
                continue
            for j in range(L):
                jj = idx_arr[j]
                if jj is None:
                    continue
                W[i, j] = self.matrix[ii, jj]
        return W, missing

    def attraction_matrix_for_sequence(
        self, sequence: Iterable[str],
    ) -> Tuple[np.ndarray, List[int], bool]:
        """Build an (L, L) per-position ATTRACTION matrix s_ij >= 0.

        For sign_convention == "negative_favorable": s = max(-W, 0).
        For sign_convention == "positive_favorable": s = max( W, 0).
        For sign_convention == "unknown": s is all-zeros and the third
        return value (`available`) is False.

        Returns
        -------
        S : (L, L) float64, non-negative.
        missing : list of indices in `sequence` not present in this table.
        available : bool — False if sign_convention == "unknown".
        """
        if self.sign_convention == "unknown":
            seq = list(sequence)
            return (
                np.zeros((len(seq), len(seq)), dtype=np.float64),
                [],
                False,
            )
        W, missing = self.weight_matrix_for_sequence(sequence)
        if self.sign_convention == "negative_favorable":
            S = np.maximum(-W, 0.0)
        else:  # positive_favorable
            S = np.maximum(W, 0.0)
        return S, missing, True


# ---------------------------------------------------------------------- #
# loader                                                                 #
# ---------------------------------------------------------------------- #

def load_mj_table_from_file(
    path: str,
    sign_convention: str = "unknown",
) -> MJContactTable:
    """Parse an MJ-style matrix file.

    The file is expected to have a one-letter-code header line and a
    square block of floats below it. The block may be lower- or
    upper-triangular with zeros on the unwritten side; the loader
    symmetrizes by max(M, M.T) of the absolute values when zero entries
    are present opposite a non-zero entry. (For the canonical project
    file, the upper triangle holds the data.)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"MJ matrix file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    if not lines:
        raise ValueError(f"empty MJ matrix file: {path}")

    header = lines[0].split()
    n = len(header)
    if n < 2:
        raise ValueError(f"MJ header has < 2 residues: {header}")

    body = lines[1:]
    if len(body) != n:
        raise ValueError(
            f"MJ matrix expects {n} body rows for {n} residues, "
            f"got {len(body)} rows"
        )

    M = np.zeros((n, n), dtype=np.float64)
    for r, ln in enumerate(body):
        toks = ln.split()
        if len(toks) != n:
            raise ValueError(
                f"row {r} has {len(toks)} entries, expected {n}"
            )
        for c, t in enumerate(toks):
            M[r, c] = float(t)

    # Symmetrize: zeros opposite a non-zero entry are placeholders, NOT
    # real "zero interaction" values. Copy the non-zero side onto the
    # placeholder side. Diagonal entries are preserved verbatim (they
    # come straight from the file). If both sides are filled and
    # disagree, keep the average (defensive — does not happen for the
    # canonical project file).
    M_sym = M.copy()
    for i in range(n):
        for j in range(i + 1, n):
            a, b = M[i, j], M[j, i]
            if a != 0.0 and b == 0.0:
                M_sym[j, i] = a
            elif b != 0.0 and a == 0.0:
                M_sym[i, j] = b
            elif a != 0.0 and b != 0.0 and abs(a - b) > 1e-9:
                v = 0.5 * (a + b)
                M_sym[i, j] = v
                M_sym[j, i] = v

    return MJContactTable(
        residues=[aa.upper() for aa in header],
        matrix=M_sym,
        sign_convention=sign_convention,
        source_path=os.path.abspath(path),
        notes={
            "n_residues": str(n),
            "all_nonpositive": str(bool(np.all(M_sym <= 0.0))),
            "max_value": f"{float(M_sym.max()):.6e}",
            "min_value": f"{float(M_sym.min()):.6e}",
        },
    )


def load_mj_table_default() -> MJContactTable:
    """Load the canonical project MJ table at ras_folding/mj_matrix.txt.

    Sign convention is set to "negative_favorable" because:
      1. The file's data is non-positive throughout (verified at load time).
      2. It matches the standard MJ convention used in the literature
         (Miyazawa & Jernigan 1996), where lower (more negative) means
         more favorable contact.
    """
    table = load_mj_table_from_file(
        _DEFAULT_MJ_PATH, sign_convention="negative_favorable",
    )
    # Defensive sanity check: if any positive entry slipped in, downgrade
    # the sign_convention to "unknown" rather than silently mislabel it.
    if not bool(np.all(table.matrix <= 1e-9)):
        table.sign_convention = "unknown"
        table.notes["sign_downgraded_reason"] = (
            "positive entries present; manual sign verification needed"
        )
    return table


__all__ = [
    "MJContactTable",
    "load_mj_table_from_file",
    "load_mj_table_default",
]
