from typing import Optional, Tuple

import ase.neighborlist
import numpy as np


def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pbc is None:
        pbc = (False, False, False)

    if cell is None or (cell == 0.0).all():
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)

    # Note (mario): I swapped senders and receivers here
    # j = senders, i = receivers instead of the other way around
    # such that the receivers are always in the central cell.
    # This is important to propagate message passing towards the center which can be useful in some cases.
    receivers, senders, senders_unit_shifts = ase.neighborlist.primitive_neighbor_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
        self_interaction=True,  # we want edges from atom to itself in different periodic images
        use_scaled_positions=False,  # positions are not scaled positions
    )

    if not true_self_interaction:
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = senders == receivers
        true_self_edge &= np.all(senders_unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        senders = senders[keep_edge]
        receivers = receivers[keep_edge]
        senders_unit_shifts = senders_unit_shifts[keep_edge]

    # Build output
    edge_index = np.stack((senders, receivers))  # [2, n_edges]

    # From the docs: With the shift vector S, the distances D between atoms can be computed from
    # D = positions[j]-positions[i]+S.dot(cell)
    # Note (mario): this is done in the function get_edge_relative_vectors
    return edge_index, senders_unit_shifts, cell
