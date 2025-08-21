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
    Greedy initial assignment using a contiguity-first, seed-then-grow approach.
    """
    logging.info("Step 3 of 5: Generating initial district map...")
    
    districts = [set() for _ in range(D)]
    pop_per_district = np.zeros(D)
    pop_tol = ideal_pop * pop_tolerance_ratio
    max_pop = ideal_pop + pop_tol
    
    remaining_blocks = get_sweep_order(gdf)
    block_coords_map = {row.GEOID20: (row.x, row.y) for row in gdf.itertuples()}

    # --- 1. SEEDING PHASE ---
    logging.info(f"Seeding {D} districts...")
    seed_indices = np.linspace(0, len(remaining_blocks) - 1, D, dtype=int)
    seeds = [remaining_blocks[i] for i in seed_indices]

    for i in range(D):
        seed_block = seeds[i]
        districts[i].add(seed_block)
        pop_per_district[i] += G.nodes[seed_block]["pop"]
    
    for i in sorted(seed_indices, reverse=True):
        del remaining_blocks[i]
    
    logging.info("Seeding complete. Starting growth phase...")
    # --- 2. GROWTH PHASE ---
    total_to_assign = len(remaining_blocks)
    for i, block in enumerate(remaining_blocks):
        block_pop = int(G.nodes[block]["pop"])

        # Find all adjacent districts
        adj_districts = []
        for j in range(D):
            if any(G.has_edge(block, b) for b in districts[j]):
                adj_districts.append((pop_per_district[j], j))

        if adj_districts:
            # From the adjacent districts, find the ones that are not full
            under_limit_neighbors = [d for d in adj_districts if d[0] + block_pop <= max_pop]

            if under_limit_neighbors:
                # If there are non-full neighbors, pick the one with the smallest population
                under_limit_neighbors.sort()
                best_idx = under_limit_neighbors[0][1]
            else:
                # If all adjacent neighbors are full, relax the pop constraint to maintain contiguity
                # and pick the one with the smallest population.
                adj_districts.sort()
                best_idx = adj_districts[0][1]
        else:
            # Fallback: True island. Use the geographic fallback.
            logging.warning(f"Block {block} is an island; using GEOGRAPHIC fallback.")
            district_centroids = {}
            for dist_idx, block_set in enumerate(districts):
                if block_set:
                    coords = [block_coords_map[b] for b in block_set]
                    district_centroids[dist_idx] = (sum(c[0] for c in coords) / len(coords), sum(c[1] for c in coords) / len(coords))
            
            block_coords = block_coords_map[block]
            best_idx = find_closest_district(block_coords, district_centroids) if district_centroids else 0

        districts[best_idx].add(block)
        pop_per_district[best_idx] += block_pop

        if (i + 1) % 5000 == 0 or i == total_to_assign - 1:
            logging.info(f"Assigned {i + 1} of {total_to_assign} blocks.")

    return districts
