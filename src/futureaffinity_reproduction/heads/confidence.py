"""A real lDDT calculation (not a stand-in) -- the same metric AlphaFold's pLDDT is trained
to predict, computed directly from two sets of coordinates so it can serve as a genuine
training target. See FutureAffinity_PyTorch's `model/heads/confidence.py:compute_lddt` for the
tensorized version of the exact same formula this reproduces on plain tuples.
"""

from __future__ import annotations

import math

from futureaffinity_reproduction.generator.tiny import AtomCoordinate

_DEFAULT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


def _distance(a: AtomCoordinate, b: AtomCoordinate) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def compute_lddt(
    predicted: tuple[AtomCoordinate, ...],
    true: tuple[AtomCoordinate, ...],
    radius: float = 15.0,
    thresholds: tuple[float, ...] = _DEFAULT_THRESHOLDS,
) -> tuple[float, ...]:
    """Per-atom lDDT (0-100): the fraction of nearby-pair distances preserved within tolerance."""
    n = len(true)
    true_distances = [[_distance(true[i], true[j]) for j in range(n)] for i in range(n)]
    predicted_distances = [[_distance(predicted[i], predicted[j]) for j in range(n)] for i in range(n)]

    scores = []
    for i in range(n):
        preserved_total = 0.0
        neighbor_count = 0
        for j in range(n):
            if i == j or true_distances[i][j] >= radius:
                continue
            neighbor_count += 1
            diff = abs(predicted_distances[i][j] - true_distances[i][j])
            preserved_total += sum(1.0 for threshold in thresholds if diff < threshold) / len(thresholds)
        scores.append((preserved_total / neighbor_count * 100.0) if neighbor_count else 100.0)
    return tuple(scores)
