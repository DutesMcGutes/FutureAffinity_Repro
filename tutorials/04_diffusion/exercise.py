from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import random

Point = Tuple[float, float, float]


@dataclass(frozen=True)
class Atom:
    index: int
    x: float
    y: float
    z: float


def target_atoms(count: int = 5) -> Tuple[Atom, ...]:
    """A tiny stand-in "clean structure": points on a spiral."""
    import math

    atoms = []
    for i in range(count):
        angle = i * 1.2
        atoms.append(Atom(index=i, x=2.0 * math.cos(angle), y=2.0 * math.sin(angle), z=i * 1.5))
    return tuple(atoms)


def noise_schedule(num_steps: int, sigma_max: float = 4.0) -> Tuple[float, ...]:
    """TODO: Produce noise levels that shrink linearly from sigma_max toward 0."""
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1.")
    return tuple(sigma_max * (1.0 - step / num_steps) for step in range(num_steps))


def add_noise(atoms: Tuple[Atom, ...], sigma: float, seed: int) -> Tuple[Atom, ...]:
    """TODO: Perturb every atom with independent Gaussian noise (the forward process)."""
    rng = random.Random(seed)
    return tuple(
        Atom(index=atom.index, x=atom.x + rng.gauss(0.0, sigma), y=atom.y + rng.gauss(0.0, sigma), z=atom.z + rng.gauss(0.0, sigma))
        for atom in atoms
    )


def denoise_step(current: Tuple[Atom, ...], target: Tuple[Atom, ...], sigma: float, sigma_next: float) -> Tuple[Atom, ...]:
    """TODO: Step from the current noisy atoms toward the target (the reverse process).

    A trained model would predict the target from the noisy input; here the
    target is already known, so the exercise focuses on the sampling loop
    shape rather than the prediction itself.
    """
    if sigma <= 0.0:
        return target
    step_fraction = 1.0 - sigma_next / sigma
    return tuple(
        Atom(
            index=cur.index,
            x=cur.x + step_fraction * (tgt.x - cur.x),
            y=cur.y + step_fraction * (tgt.y - cur.y),
            z=cur.z + step_fraction * (tgt.z - cur.z),
        )
        for cur, tgt in zip(current, target)
    )


def run_diffusion(target: Tuple[Atom, ...], seed: int = 1, num_steps: int = 8) -> Tuple[Atom, ...]:
    """TODO: Chain the schedule, forward noise, and reverse steps into one sampler."""
    schedule = noise_schedule(num_steps)
    current = add_noise(target, schedule[0], seed)
    for index, sigma in enumerate(schedule):
        sigma_next = schedule[index + 1] if index + 1 < len(schedule) else 0.0
        current = denoise_step(current, target, sigma, sigma_next)
    return current


def run_diffusion_ensemble(
    target: Tuple[Atom, ...], seeds: Tuple[int, ...] = (1, 2, 3, 4, 5), num_steps: int = 8
) -> Tuple[Tuple[Atom, ...], ...]:
    """TODO: Run the sampler once per seed -- an ensemble of independently-noised trajectories.

    Every run denoises toward the same `target`, so with a perfect denoiser
    every sample would converge exactly to it (see how `max_error` in `main`
    goes to ~0). A real (imperfect, learned) denoiser wouldn't converge
    exactly -- and the *spread* across an ensemble like this is a real,
    useful uncertainty signal, which is exactly what
    `DiffusionModule.sample_ensemble` in FutureAffinity_PyTorch is for.
    """
    return tuple(run_diffusion(target, seed=seed, num_steps=num_steps) for seed in seeds)


def per_atom_uncertainty(ensemble: Tuple[Tuple[Atom, ...], ...]) -> Tuple[float, ...]:
    """TODO: Per-atom RMS spread across the ensemble."""
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


def main() -> None:
    target = target_atoms()
    result = run_diffusion(target, seed=1, num_steps=8)
    max_error = max(abs(r.x - t.x) + abs(r.y - t.y) + abs(r.z - t.z) for r, t in zip(result, target))
    print(f"atoms={len(result)} max_error_after_denoising={max_error:.6f}")

    ensemble = run_diffusion_ensemble(target, seeds=(1, 2, 3, 4, 5), num_steps=8)
    uncertainty = per_atom_uncertainty(ensemble)
    print(f"ensemble_samples={len(ensemble)} mean_uncertainty={sum(uncertainty) / len(uncertainty):.6f}")


if __name__ == "__main__":
    main()
