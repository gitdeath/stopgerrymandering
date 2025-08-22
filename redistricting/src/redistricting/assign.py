from __future__ import annotations

import logging
import pandas as pd
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
    Builds districts simultaneously using an inertia-guided, competitive region-growing algorithm.
    """
    logging.info("Step 3 of 5: Generating initial district map using Competitive Growth...")
    
    districts = [set() for _ in range(D)]
    unassigned_blocks = set(gdf["GEOID20"])
    total_blocks = len(unassigned_blocks) # Get total block count once at the start
    max_pop = ideal_pop * (1 + pop_tolerance_ratio)
    
    # Pre-calculate data for performance
    block_pop_map = {row.GEOID20: int(row.P1_001N) for row in gdf.itertuples()}
    block_coords_map = {row.GEOID20: (row.x, row.y) for row in gdf.itertuples()}
    
    # --- 1. SEEDING PHASE ---
    logging.info(f"Seeding {D} districts...")
    populated_gdf = gdf[gdf["P1_001N"] > 0]
    sorted_populated = populated_gdf.sort_values(by=["y", "x"], ascending=[False, True])
    
    if len(sorted_populated) < D:
        raise ValueError("Not enough populated blocks to seed all districts.")
        
    seed_indices = np.linspace(0, len(sorted_populated) - 1, D, dtype=int)
    seeds = [sorted_populated.iloc[i]["GEOID20"] for i in seed_indices]

    for i, seed_block in enumerate(seeds):
        districts[i].add(seed_block)
        unassigned_blocks.remove(seed_block)
    
    pop_per_district = np.array([sum(block_pop_map.get(b,0) for b in d) for d in districts])
    
    logging.info("Seeding complete. Starting competitive growth phase...")
    # --- 2. COMPETITIVE GROWTH PHASE ---
    districts_at_max_pop = [False] * D
    
    while unassigned_blocks:
        # Check every N blocks for progress logging
        blocks_assigned = total_blocks - len(unassigned_blocks)
        if blocks_assigned > 0 and blocks_assigned % 5000 < D:
             logging.info(f"Assigned {blocks_assigned} / {total_blocks} blocks...")

        for i in range(D):
            if districts_at_max_pop[i]:
                continue

            frontier = {
                neighbor
                for block in districts[i]
                for neighbor in G.neighbors(block)
                if neighbor in unassigned_blocks
            }

            if not frontier:
                continue

            current_pop = pop_per_district[i]
            if current_pop == 0: continue
            
            pop_x = sum(block_coords_map[b][0] * block_pop_map[b] for b in districts[i])
            pop_y = sum(block_coords_map[b][1] * block_pop_map[b] for b in districts[i])
            centroid_x, centroid_y = pop_x / current_pop, pop_y / current_pop
            
            best_block_to_add = None
            lowest_inertia_gain = float('inf')

            for frontier_block in frontier:
                pop = block_pop_map[frontier_block]
                coords = block_coords_map[frontier_block]
                dist_sq = (coords[0] - centroid_x)**2 + (coords[1] - centroid_y)**2
                inertia_gain = pop * dist_sq
                
                if inertia_gain < lowest_inertia_gain:
                    lowest_inertia_gain = inertia_gain
                    best_block_to_add = frontier_block
            
            if best_block_to_add:
                pop_to_add = block_pop_map[best_block_to_add]
                
                districts[i].add(best_block_to_add)
                unassigned_blocks.remove(best_block_to_add)
                pop_per_district[i] += pop_to_add

                if pop_per_district[i] >= max_pop:
                    districts_at_max_pop[i] = True
                    logging.info(f"District {i+1} has reached its population cap.")

    logging.info("Initial assignment complete.")
    return [list(d) for d in districts]
