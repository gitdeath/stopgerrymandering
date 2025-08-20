from __future__ import annotations

import hashlib
import logging
import numpy as np
import pandas as pd


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


def find_closest_district(block_coords, district_centroids):
    """Finds the district centroid closest to a given block."""
    closest_dist = float('inf')
    best_idx = -1
    for idx, centroid in district_centroids.items():
        dist = (block_coords[0] - centroid[0])**2 + (block_coords[1] - centroid[1])**2
        if dist < closest_dist:
            closest_dist = dist
            best_idx = idx
    return best_idx


def initial_assignment(gdf, G, D: int, ideal_pop: float, pop_tolerance_ratio: float):
    """
    Greedy initial assignment of blocks to districts with adjacency + pop guardrails.
    Uses a hybrid seed-then-grow approach.
    """
    logging.info("Step 3 of 5: Generating initial district map...")
    
    districts = [set() for _ in range(D)]
    pop_per_district = np.zeros(D)
    pop_tol = ideal_pop * pop_tolerance_ratio
    
    # Create a mutable list of all blocks to be assigned
    remaining_blocks = get_sweep_order(gdf)
    
    # Pre-calculate block coordinates for quick lookups
    block_coords_map = {row.GEOID20: (row.x, row.y) for row in gdf.itertuples()}

    # --- 1. SEEDING PHASE ---
    # Deterministically pick D seed blocks spread evenly across the sweep order.
    logging.info(f"Seeding {D} districts...")
    seed_indices = np.linspace(0, len(remaining_blocks) - 1, D, dtype=int)
    seeds = [remaining_blocks[i] for i in seed_indices]

    for i in range(D):
        seed_block = seeds[i]
        districts[i].add(seed_block)
        pop_per_district[i] += G.nodes[seed_block]["pop"]
    
    # Remove the seed blocks from the list of blocks to assign
    # Iterate backwards to avoid index shifting issues
    for i in sorted(seed_indices, reverse=True):
        del remaining_blocks[i]
    
    logging.info("Seeding complete. Starting growth phase...")
    # --- 2. GROWTH PHASE ---
    total_to_assign = len(remaining_blocks)
    for i, block in enumerate(remaining_blocks):
        block_pop = int(G.nodes[block]["pop"])

        candidates = []
        for j in range(D):
            # A district is a candidate if the block is adjacent and pop is within tolerance
            if any(G.has_edge(block, b) for b in districts[j]):
                if (pop_per_district[j] + block_pop <= ideal_pop + pop_tol):
                    candidates.append((pop_per_district[j], j))

        if candidates:
            candidates.sort() # Prefer smallest population
            best_idx = candidates[0][1]
        else:
            # Fallback 1: No ideal candidates. Relax pop constraint, require adjacency.
            adj_candidates = []
            for j in range(D):
                if any(G.has_edge(block, b) for b in districts[j]):
                    adj_candidates.append((pop_per_district[j], j))

            if adj_candidates:
                adj_candidates.sort()
                best_idx = adj_candidates[0][1]
            else:
                # Fallback 2: True island. Use geographic fallback.
                logging.warning(f"Block {block} is an island; using GEOGRAPHIC fallback.")
                district_centroids = {}
                for dist_idx, block_set in enumerate(districts):
                    if block_set:
                        coords = [block_coords_map[b] for b in block_set]
                        centroid_x = sum(c[0] for c in coords) / len(coords)
                        centroid_y = sum(c[1] for c in coords) / len(coords)
                        district_centroids[dist_idx] = (centroid_x, centroid_y)
                
                block_coords = block_coords_map[block]
                best_idx = find_closest_district(block_coords, district_centroids)

        districts[best_idx].add(block)
        pop_per_district[best_idx] += block_pop

        if (i + 1) % 1000 == 0 or i == total_to_assign -1:
            logging.info(f"Assigned {i + 1} of {total_to_assign} blocks.")

    return districts
