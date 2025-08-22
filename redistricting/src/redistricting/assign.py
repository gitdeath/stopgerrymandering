from __future__ import annotations

import logging
import pandas as pd
import numpy as np


def initial_assignment(gdf, G, D: int, ideal_pop: float, pop_tolerance_ratio: float):
    """
    Builds districts simultaneously using an inertia-guided, competitive region-growing algorithm.
    """
    logging.info("Step 3 of 5: Generating initial district map using Competitive Growth...")
    
    districts = [set() for _ in range(D)]
    unassigned_blocks = set(gdf["GEOID20"])
    max_pop = ideal_pop * (1 + pop_tolerance_ratio)
    
    # Pre-calculate data for performance
    block_pop_map = {row.GEOID20: int(row.P1_001N) for row in gdf.itertuples()}
    block_coords_map = {row.GEOID20: (row.x, row.y) for row in gdf.itertuples()}
    
    # --- 1. SEEDING PHASE ---
    logging.info(f"Seeding {D} districts...")
    # Find seeds only from populated blocks
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
        for i in range(D):
            if districts_at_max_pop[i]:
                continue

            # Find the frontier for the current district
            frontier = {
                neighbor
                for block in districts[i]
                for neighbor in G.neighbors(block)
                if neighbor in unassigned_blocks
            }

            if not frontier:
                continue # Skip if this district can't expand

            # --- Inertia Heuristic ---
            # Calculate current centroid
            current_pop = pop_per_district[i]
            if current_pop == 0: continue
            
            pop_x = sum(block_coords_map[b][0] * block_pop_map[b] for b in districts[i])
            pop_y = sum(block_coords_map[b][1] * block_pop_map[b] for b in districts[i])
            centroid_x, centroid_y = pop_x / current_pop, pop_y / current_pop
            
            # Find the best block on the frontier to add to this district
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
                
                # Add the block and update state
                districts[i].add(best_block_to_add)
                unassigned_blocks.remove(best_block_to_add)
                pop_per_district[i] += pop_to_add

                # Check if this district is now full
                if pop_per_district[i] >= max_pop:
                    districts_at_max_pop[i] = True
                    logging.info(f"District {i+1} has reached its population cap.")

        if (len(all_blocks) - len(unassigned_blocks)) % 5000 < D:
             logging.info(f"Assigned {len(all_blocks) - len(unassigned_blocks)} / {len(all_blocks)} blocks...")
             
    logging.info("Initial assignment complete.")
    return [list(d) for d in districts]
