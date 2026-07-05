# Chapter 06: Synthetic Supervision

Real affinity measurements are scarce (maybe 10^5-10^6 high-quality ones exist, total). Structures
and sequences number in the hundreds of millions. This chapter builds the piece that bridges that
gap: a cheap, dependency-free "docking engine" that generates plausible poses and energies at
whatever scale you want.

`toy_pairwise_energy` is a Lennard-Jones-style formula -- not a real force field, but physics
*shaped*: a steep repulsive wall at very short distances, an attractive well at a moderate
distance. `dock` generates several random rigid-body poses of a ligand around a receptor and
returns the lowest-energy ones. Neither is trained; both are infinitely repeatable and free.

The idea this teaches: pretrain an affinity head on millions of these noisy, synthetic
(pose, energy) pairs first, so the network learns approximate physics before ever seeing a real
measurement -- then fine-tune on the scarce real data. See FutureAffinity_PyTorch's
`datasources/mock_docking.py` for the trainable-model version of this same idea, and
`vina_adapter.py` / `openmm_adapter.py` for where real physics (AutoDock Vina, OpenMM) would plug
into the exact same interface once you have real molecule files to feed them.

## Student Task

Open `exercise.py` and study the TODO-annotated functions:

1. `toy_pairwise_energy` -- the Lennard-Jones-style toy energy between a receptor and a ligand pose.
2. `dock` -- generate random poses and return the lowest-energy ones.

Then run:

```bash
python tutorials/06_synthetic_supervision/exercise.py
```

To compare against the reference implementation:

```bash
python scripts/generate_test_results.py 06_synthetic_supervision
```
