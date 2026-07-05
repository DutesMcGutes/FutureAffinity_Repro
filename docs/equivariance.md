# Equivariance: why orientation shouldn't matter, and how to make sure it doesn't

A structure predictor should not care where in space, or facing which way, its input happens to be.
Rotate the inputs and the predicted structure should rotate with them (equivariance); translate the
inputs and the prediction should translate with them. If a model's output *changes* under a rotation
that shouldn't matter, it's wasting capacity learning to be robust to something it should get for
free.

There are two ways to get this property:

1. **Build it into the architecture** — SE(3)-equivariant layers (tensor-field networks, EGNN) that
   are equivariant by construction. Exact, sample-efficient, but more complex and historically
   slower.
2. **Instill it through data** — use a plain (non-equivariant) network, but (a) *center* every
   structure so absolute position carries no signal, and (b) *randomly rotate* every training
   example so the network sees all orientations and learns not to prefer any. Approximate and
   learned, but simple and scalable.

AlphaFold3 (and the `FutureAffinity_PyTorch` implementation this tutorial mirrors) takes the second
route: centering makes translation-invariance *exact*, and random SO(3) rotation augmentation makes
rotation-robustness something the model learns.

## See it yourself

Chapter 08 makes this concrete: it shows that a **naive coordinate MSE** between a structure and a
rotated copy of itself blows up (the coordinates all moved), while a **superposition-free metric
like lDDT** stays at 100 (the internal geometry didn't change). That contrast is the whole idea:
what matters is internal geometry, not the arbitrary frame — so both your loss/augmentation strategy
*and* your evaluation metrics have to respect it.

See `docs/adr/0002` in the paired `FutureAffinity_PyTorch` repo for the decision record, and its
`geometry.py` / `tests/test_equivariance.py` for the real, tested implementation.
