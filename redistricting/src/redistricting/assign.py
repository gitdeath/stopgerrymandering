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
    Greedy initial assignment using a geographically aware, "right of first refusal" heuristic.
    """
    logging.info("Step 3 of 5: Generating initial district map...")
    
    districts = [set() for _ in range(D)]
    pop_per_district = np.zeros(D)
    max_pop = ideal_pop * (1 + pop_tolerance_ratio)
    
    all_blocks = get_sweep_order(gdf)
    block_coords_map = {row.GEOID20: (row.x, row.y) for row in gdf.itertuples()}

    # --- 1. SEEDING AND LOYALTY MAPPING ---
    logging.info(f"Seeding {D} districts and determining block loyalties...")
    seed_indices = np.linspace(0, len(all_blocks) - 1, D, dtype=int)
    seeds = [all_blocks[i] for i in seed_indices]
    seed_coords = {i: block_coords_map[seed] for i, seed in enumerate(seeds)}

    block_loyalty = {
        block: find_closest_district(coords, seed_coords)
        for block, coords in block_coords_map.items()
    }

    for i, seed_block in enumerate(seeds):
        districts[i].add(seed_block)
        pop_per_district[i] += G.nodes[seed_block]["pop"]
    
    remaining_blocks = [b for b in all_blocks if b not in seeds]
    
    logging.info("Seeding complete. Starting geographically-aware growth phase...")
    # --- 2. GROWTH PHASE ---
    total_to_assign = len(remaining_blocks)
    for i, block in enumerate(remaining_blocks):
        block_pop = int(G.nodes[block]["pop"])

        adj_districts = {j for j in range(D) if any(G.has_edge(block, b) for b in districts[j])}

        if adj_districts:
            home_district = block_loyalty[block]
            
            under_limit_neighbors = {
                j for j in adj_districts if pop_per_district[j] + block_pop <= max_pop
            }

            best_idx = -1
            if home_district in under_limit_neighbors:
                # "Right of First Refusal": Assign to home district if it's a valid option.
                best_idx = home_district
            elif under_limit_neighbors:
                # Home district is not a valid option; block becomes a "free agent."
                # Assign to the least-populated valid neighbor.
                best_idx = min(under_limit_neighbors, key=lambda d: pop_per_district[d])
            else:
                # All neighbors are full; relax pop constraint and assign to least-populated neighbor.
                best_idx = min(adj_districts, key=lambda d: pop_per_district[d])
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
