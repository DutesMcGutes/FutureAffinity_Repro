"""A real (distance-thresholded) contact map -- no learned weights, but a genuine geometric
calculation, exactly the target `heads/contacts.py:ContactHead` in FutureAffinity_PyTorch is trained
to predict from the pair representation alone (i.e. without looking at coordinates directly).
"""

from __future__ import annotations

import math

from futureaffinity_reproduction.generator.tiny import AtomCoordinate

DEFAULT_CONTACT_THRESHOLD = 8.0


def contact_map(
    coordinates: tuple[AtomCoordinate, ...], threshold: float = DEFAULT_CONTACT_THRESHOLD
) -> tuple[tuple[int, ...], ...]:
    n = len(coordinates)
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            a, b = coordinates[i], coordinates[j]
            distance = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
            row.append(1 if distance < threshold else 0)
        rows.append(tuple(row))
    return tuple(rows)
