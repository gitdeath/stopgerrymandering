from __future__ import annotations

import logging
import pandas as pd
import numpy as np

# Removed the direct import of compute_inertia as we are using a faster, inline heuristic

def initial_assignment(gdf, G, D: int, ideal_pop: float, pop_tolerance_ratio: float):
    """
    Builds districts one at a time using a faster, inertia-guided growth algorithm.
    """
    logging.info("Step 3 of 5: Generating initial district map using Inertia-Guided Growth...")
    
    districts = [set() for _ in range(D)]
    unassigned_blocks = set(gdf["GEOID20"])
    min_pop = ideal_pop * (1 - pop_tolerance_ratio)
    
    # Pre-calculate data for performance
    block_pop_map = {row.GEOID20: int(row.P1_001N) for row in gdf.itertuples()}
    block_coords_map = {row.GEOID20: (row.x, row.y) for row in gdf.itertuples()}
    
    for i in range(D - 1):
        logging.info(f"--- Building District {i+1}/{D} ---")
        
        # --- 1. Find the Seed Block (Cardinal Direction) ---
        remaining_gdf = gdf[gdf["GEOID20"].isin(unassigned_blocks)]
        sorted_remaining = remaining_gdf.sort_values(by=["y", "x"], ascending=[False, True])
        
        if sorted_remaining.empty:
            logging.warning(f"No more blocks to assign, stopping at District {i}.")
            break
            
        seed_block = sorted_remaining.iloc[0]["GEOID20"]
        
        # --- 2. Grow the District ---
        current_district_blocks = {seed_block}
        unassigned_blocks.remove(seed_block)
        
        current_pop = block_pop_map[seed_block]
        # Keep track of weighted sums for fast centroid calculation
        current_pop_x = block_coords_map[seed_block][0] * current_pop
        current_pop_y = block_coords_map[seed_block][1] * current_pop

        while current_pop < min_pop:
            frontier = {
                neighbor
                for block in current_district_blocks
                for neighbor in G.neighbors(block)
                if neighbor in unassigned_blocks
            }

            if not frontier:
                logging.warning(f"District {i+1} ran out of neighbors before reaching target pop.")
                break

            # --- FASTER HEURISTIC ---
            # Calculate the current district's centroid once
            centroid_x = current_pop_x / current_pop
            centroid_y = current_pop_y / current_pop

            best_block_to_add = None
            lowest_inertia_gain = float('inf')

            for frontier_block in frontier:
                pop = block_pop_map[frontier_block]
                coords = block_coords_map[frontier_block]
                
                # Estimate inertia gain relative to the *current* centroid. This is a fast proxy.
                dist_sq = (coords[0] - centroid_x)**2 + (coords[1] - centroid_y)**2
                inertia_gain = pop * dist_sq
                
                if inertia_gain < lowest_inertia_gain:
                    lowest_inertia_gain = inertia_gain
                    best_block_to_add = frontier_block
            
            if best_block_to_add:
                pop_to_add = block_pop_map[best_block_to_add]
                coords_to_add = block_coords_map[best_block_to_add]

                current_district_blocks.add(best_block_to_add)
                unassigned_blocks.remove(best_block_to_add)
                
                # Update population and weighted sums for next centroid calculation
                current_pop += pop_to_add
                current_pop_x += coords_to_add[0] * pop_to_add
                current_pop_y += coords_to_add[1] * pop_to_add

                # --- NEW PROGRESS LOGGING ---
                if len(current_district_blocks) % 2000 == 0:
                    logging.info(f"   ... District {i+1} has grown to {len(current_district_blocks)} blocks, "
                                 f"pop: {current_pop:,}")
            else:
                logging.warning(f"Could not find a best block for District {i+1}, growth stalled.")
                break
        
        districts[i] = current_district_blocks
        logging.info(f"District {i+1} complete with population {current_pop:,}.")

    # --- 3. The Final District ---
    final_district_idx = D - 1
    districts[final_district_idx] = unassigned_blocks
    final_pop = sum(block_pop_map.get(b, 0) for b in unassigned_blocks)
    logging.info(f"--- Building District {D}/{D} ---")
    logging.info(f"District {D} complete with remaining {len(unassigned_blocks)} blocks and population {final_pop:,}.")

    return [list(d) for d in districts]
