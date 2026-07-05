# Metrics: what "accurate" means, and how each one lies

A structure/affinity model is only as trustworthy as the metric you judge it by, and every metric
has a blind spot. A short field guide (the real, tensorized versions live in
`FutureAffinity_PyTorch/evaluation/metrics.py`; Chapter 05 here implements lDDT and contacts in
pure Python).

## Structure

- **RMSD** — average atom displacement after optimal alignment. Intuitive, but one badly-placed loop
  dominates it, and it's undefined across different-length structures.
- **lDDT** — fraction of *local* distances preserved. Superposition-free, so it can't be fooled by a
  bad global alignment, and it's what confidence (pLDDT) is trained to predict. Chapter 08 uses its
  rotation-invariance to make the equivariance point.
- **lDDT-PLI** — lDDT restricted to protein-ligand interface contacts. The headline number for
  co-folding: did the model get the *interaction* right, regardless of the protein/ligand alone.
- **TM-score** — length-normalized global similarity in (0, 1]; >0.5 ≈ "same fold."
- **DockQ** — one [0, 1] score combining interface-contact recall with interface/ligand RMSD; the
  standard for ranking predicted complexes.

## Affinity

- **Pearson r** — linear correlation; cares about getting the scale right.
- **Spearman ρ** — rank correlation; often what matters most, because prioritizing compounds is a
  *ranking* problem. A model can rank well (high ρ) while mis-estimating absolute Kd.
- **RMSE / MAE** — absolute error, for when you need to act on the number ("is this sub-micromolar?").

## The meta-point

The number is meaningless without (a) a leakage-controlled split (see docs/splits-and-leakage.md)
and (b) knowing the metric's blind spot. "0.9 Pearson" on a random split of a single congeneric
series says almost nothing; "0.5 Spearman on a scaffold split" might be genuinely useful.
