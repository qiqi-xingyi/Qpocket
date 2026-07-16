# Author: Yuqi Zhang
"""Physical / geometric constants used throughout ras_folding."""

# CA-CA virtual bond length (Å). Standard peptide bond.
CA_CA_LENGTH: float = 3.80

# vdW-ish radii used in geometric exclusion checks
R_ENV_CA: float = 2.5          # environment CA exclusion radius
R_LIG_HEAVY: float = 1.8       # ligand heavy atom vdW
R_FRAG_CA: float = 1.8         # fragment CA atomic radius

# Ligand contact shell (Å)
R_LIG_CONTACT_INNER: float = 3.5
R_LIG_CONTACT_OUTER: float = 5.5

# Env proximity clip
R_ENV_PROX_MAX: float = 10.0
