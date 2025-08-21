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
        dist_sq = (block_coords[0] - centroid[0])**2 + (block_coords[1] - centroid[1])**2
        if dist_sq < closest_dist:
            closest_dist = dist_sq
            best_idx = idx
    return best_idx


def get_district_centroids(districts, block_coords_map, block_pop_map):
    """Calculates the population-weighted centroid for each district."""
    centroids = {}
    for i, d in enumerate(districts):
        if not d: continue
        
        total_pop = sum(block_pop_map[b] for b in d)
        if total_pop == 0: continue

        cx = sum(block_coords_map[b][0] * block_pop_map[b] for b in d) / total_pop
        cy = sum(block_coords_map[b][1] * block_pop_map[b] for b in d) / total_pop
        centroids[i] = (cx, cy)
    return centroids


def initial_assignment(gdf, G, D: int, ideal_pop: float, pop_tolerance_ratio: float):
    """
    Greedy initial assignment using a cost function that balances population and compactness (inertia).
    """
    logging.info("Step 3 of 5: Generating initial district map...")
    
    districts = [set() for _ in range(D)]
    pop_per_district = np.zeros(D)
    max_pop = ideal_pop * (1 + pop_tolerance_ratio)
    
    all_blocks = get_sweep_order(gdf)
    block_coords_map = {row.GEOID20: (row.x, row.y) for row in gdf.itertuples()}
    block_pop_map = {row.GEOID20: row.P1_001N for row in gdf.itertuples()}

    # --- 1. SEEDING PHASE ---
    logging.info(f"Seeding {D} districts...")
    seed_indices = np.linspace(0, len(all_blocks) - 1, D, dtype=int)
    seeds = [all_blocks[i] for i in seed_indices]

    for i, seed_block in enumerate(seeds):
        districts[i].add(seed_block)
        pop_per_district[i] += G.nodes[seed_block]["pop"]
    
    remaining_blocks = [b for b in all_blocks if b not in seeds]
    
    logging.info("Seeding complete. Starting cost-function-based growth phase...")
    # --- 2. GROWTH PHASE ---
    total_to_assign = len(remaining_blocks)
    for i, block in enumerate(remaining_blocks):
        block_pop = int(G.nodes[block]["pop"])
        block_coords = block_coords_map[block]

        adj_districts = {j for j in range(D) if any(G.has_edge(block, b) for b in districts[j])}

        if adj_districts:
            candidate_districts = [j for j in adj_districts if pop_per_district[j] + block_pop <= max_pop]

            if candidate_districts:
                # --- COST FUNCTION HEURISTIC ---
                # Score candidates to find the best choice.
                candidates_with_scores = []
                # Get current centroids to estimate inertia change
                centroids = get_district_centroids(districts, block_coords_map, block_pop_map)

                for j in candidate_districts:
                    # Factor 1: Population Need (lower is better)
                    pop_ratio = pop_per_district[j] / ideal_pop
                    
                    # Factor 2: Compactness Cost (lower is better)
                    # Estimate the increase in inertia by adding this block
                    centroid_x, centroid_y = centroids.get(j, block_coords) # Use block coords if district is empty
                    inertia_cost = block_pop * ((block_coords[0] - centroid_x)**2 + (block_coords[1] - centroid_y)**2)
                    
                    # We store tuples for sorting: primary key is pop_ratio, secondary is inertia_cost
                    candidates_with_scores.append((pop_ratio, inertia_cost, j))
                
                # Sort by pop_ratio, then by inertia_cost to break ties. Lowest score is best.
                candidates_with_scores.sort()
                best_idx = candidates_with_scores[0][2]
            else:
                # All neighbors are full; relax pop constraint and assign to least-populated neighbor.
                best_idx = min(adj_districts, key=lambda d: pop_per_district[d])
        else:
            # Fallback: True island. Use the geographic fallback.
            logging.warning(f"Block {block} is an island; using GEOGRAPHIC fallback.")
            district_centroids = get_district_centroids(districts, block_coords_map, block_pop_map)
            best_idx = find_closest_district(block_coords_map[block], district_centroids) if district_centroids else 0

        districts[best_idx].add(block)
        pop_per_district[best_idx] += pop_per_district[best_idx] + block_pop

        if (i + 1) % 5000 == 0 or i == total_to_assign - 1:
            logging.info(f"Assigned {i + 1} of {total_to_assign} blocks.")

    return districts
