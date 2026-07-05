"""A tiny diffusion-style coordinate sampler.

Real diffusion denoises atom positions with a learned network conditioned
on the trunk representation (see FutureAffinity_PyTorch's `model/diffusion.py`
for the real EDM-style version). There is no learned network here: the
deterministic demo coordinates stand in for "what a trained denoiser would
predict," so this module demonstrates the same forward-noise /
reverse-denoise loop shape a trained sampler would use, end to end, without
any trained weights.

`run_diffusion_ensemble` runs that same loop from several different random
seeds -- a real structural ensemble, in miniature, teaching the same idea
`DiffusionModule.sample_ensemble` implements for real: the model's honest
output is a distribution over structures, not one point estimate, and the
spread across that distribution is a form of uncertainty "for free".
"""

from __future__ import annotations

import random

from futureaffinity_reproduction.features.builder import FeatureBatch
from futureaffinity_reproduction.generator.tiny import AtomCoordinate, generate_demo_coordinates


def noise_schedule(num_steps: int, sigma_max: float = 4.0) -> tuple[float, ...]:
    """Linearly decreasing noise levels, from sigma_max down toward 0."""
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1.")
    return tuple(sigma_max * (1.0 - step / num_steps) for step in range(num_steps))


def add_noise(coordinates: tuple[AtomCoordinate, ...], sigma: float, seed: int) -> tuple[AtomCoordinate, ...]:
    rng = random.Random(seed)
    return tuple(
        AtomCoordinate(
            atom_index=coord.atom_index,
            x=coord.x + rng.gauss(0.0, sigma),
            y=coord.y + rng.gauss(0.0, sigma),
            z=coord.z + rng.gauss(0.0, sigma),
        )
        for coord in coordinates
    )


def denoise_step(
    current: tuple[AtomCoordinate, ...],
    target: tuple[AtomCoordinate, ...],
    sigma: float,
    sigma_next: float,
) -> tuple[AtomCoordinate, ...]:
    """Move each atom part of the way from its current position toward the target.

    The step fraction grows as sigma shrinks, so the trajectory starts noisy
    and lands exactly on the target once sigma_next reaches 0. That's the
    same shape as a deterministic reverse diffusion step.
    """
    if sigma <= 0.0:
        return target
    step_fraction = 1.0 - sigma_next / sigma
    updated = []
    for cur, tgt in zip(current, target):
        updated.append(
            AtomCoordinate(
                atom_index=cur.atom_index,
                x=cur.x + step_fraction * (tgt.x - cur.x),
                y=cur.y + step_fraction * (tgt.y - cur.y),
                z=cur.z + step_fraction * (tgt.z - cur.z),
            )
        )
    return tuple(updated)


def run_diffusion(features: FeatureBatch, seed: int = 1, num_steps: int = 8) -> tuple[AtomCoordinate, ...]:
    target = generate_demo_coordinates(features, seed=seed)
    schedule = noise_schedule(num_steps)
    current = add_noise(target, schedule[0], seed)

    for index, sigma in enumerate(schedule):
        sigma_next = schedule[index + 1] if index + 1 < len(schedule) else 0.0
        current = denoise_step(current, target, sigma, sigma_next)

    return current


def run_diffusion_ensemble(
    features: FeatureBatch, seeds: tuple[int, ...] = (1, 2, 3, 4, 5), num_steps: int = 8
) -> tuple[tuple[AtomCoordinate, ...], ...]:
    """Runs `run_diffusion` once per seed -- an ensemble of independently-noised trajectories."""
    return tuple(run_diffusion(features, seed=seed, num_steps=num_steps) for seed in seeds)


def per_atom_uncertainty(ensemble: tuple[tuple[AtomCoordinate, ...], ...]) -> tuple[float, ...]:
    """Per-atom RMS spread across an ensemble -- the same idea as FutureAffinity_PyTorch's
    `heads/uncertainty.py:structural_uncertainty`, computed here on plain tuples."""
    num_atoms = len(ensemble[0])
    num_samples = len(ensemble)
    uncertainties = []
    for atom_index in range(num_atoms):
        xs = [sample[atom_index].x for sample in ensemble]
        ys = [sample[atom_index].y for sample in ensemble]
        zs = [sample[atom_index].z for sample in ensemble]
        mean_x, mean_y, mean_z = sum(xs) / num_samples, sum(ys) / num_samples, sum(zs) / num_samples
        mean_squared_deviation = sum(
            (x - mean_x) ** 2 + (y - mean_y) ** 2 + (z - mean_z) ** 2 for x, y, z in zip(xs, ys, zs)
        ) / num_samples
        uncertainties.append(mean_squared_deviation**0.5)
    return tuple(uncertainties)
