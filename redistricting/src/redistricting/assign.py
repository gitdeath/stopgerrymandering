from __future__ import annotations

import hashlib
import logging
import numpy as np


def get_sweep_order(gdf):
    """
    Deterministic sweep order based on a hash of all block IDs.
    Produces one of four sweeps over (x,y) to avoid bias.
    """
    block_ids = sorted(gdf["GEOID20"])
    hash_val = hashlib.sha256("".join(block_ids).encode()).hexdigest()
    sweep_idx = int(hash_val, 16) % 4

    if sweep_idx == 0:  # NE: descending y, ascending x
        return gdf.sort_values(by=["y", "x"], ascending=[False, True])["GEOID20"].tolist()
    elif sweep_idx == 1:  # SW: ascending y, ascending x
        return gdf.sort_values(by=["y", "x"], ascending=[True, True])["GEOID20"].tolist()
    elif sweep_idx == 2:  # SE: ascending y, descending x
        return gdf.sort_values(by=["y", "x"], ascending=[True, False])["GEOID20"].tolist()
    else:  # NW: descending y, descending x
        return gdf.sort_values(by=["y", "x"], ascending=[False, False])["GEOID20"].tolist()


def initial_assignment(gdf, G, D: int, ideal_pop: float, pop_tolerance_ratio: float):
    """
    Greedy initial assignment of blocks to districts with adjacency + pop guardrails.
    """
    logging.info("Step 3 of 5: Generating initial district map...")
    sweep_order = get_sweep_order(gdf)

    districts = [set() for _ in range(D)]
    pop_per_district = [0] * D
    pop_tol = ideal_pop * pop_tolerance_ratio

    for i, block in enumerate(sweep_order):
        block_pop = int(G.nodes[block]["pop"])

        # Candidate districts where adding this block keeps pop within tolerance
        # and the block is adjacent to at least one block already in the district
        candidates = []
        for j in range(D):
            is_adjacent = not districts[j] or any(G.has_edge(block, b) for b in districts[j])
            if (pop_per_district[j] + block_pop <= ideal_pop + pop_tol) and is_adjacent:
                candidates.append((pop_per_district[j], j))

        if candidates:
            candidates.sort()  # prefer the district with the smallest current population
            best_idx = candidates[0][1]
        else:
            # fallback: put into the currently least-populated district
            best_idx = int(np.argmin(pop_per_district))

        districts[best_idx].add(block)
        pop_per_district[best_idx] += block_pop

        if (i + 1) % 1000 == 0 or i == len(sweep_order) - 1:
            logging.info(f"Assigned {i + 1} of {len(sweep_order)} blocks.")

    return districts
