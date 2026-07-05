# Chapter 04: Diffusion

Builds a tiny diffusion-style coordinate sampler: a noise schedule, a forward-noise step, and a
reverse-denoising step, chained into `run_diffusion`. There's no learned network here -- the
target coordinates are already known, so the exercise is about the *shape* of the sampling loop
(the same shape a trained denoiser would run inside), not about prediction.

Then it takes the one step that matters for everything after this chapter: run that same sampler
several times from different random seeds (`run_diffusion_ensemble`), and look at how much the
results disagree (`per_atom_uncertainty`). With a perfect denoiser converging to a known target,
that disagreement goes to ~0 here -- but with a real, imperfect, learned denoiser (as in
FutureAffinity_PyTorch's `DiffusionModule`), it doesn't, and the spread across the ensemble becomes a
genuine, useful uncertainty estimate "for free": no separate uncertainty-prediction network
needed, just several honest samples from the same model.

## Student Task

Open `exercise.py` and study the TODO-annotated functions:

1. `noise_schedule` -- noise levels shrinking linearly from `sigma_max` to 0.
2. `add_noise` -- the forward process (independent Gaussian noise per atom).
3. `denoise_step` -- the reverse process (step toward the target as sigma shrinks).
4. `run_diffusion` -- chain the schedule and both processes into one sampler.
5. `run_diffusion_ensemble` -- run the sampler from several seeds.
6. `per_atom_uncertainty` -- the per-atom RMS spread across that ensemble.

Then run:

```bash
python tutorials/04_diffusion/exercise.py
```

To compare against the reference implementation:

```bash
python scripts/generate_test_results.py 04_diffusion
```
