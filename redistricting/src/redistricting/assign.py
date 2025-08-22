from __future__ import annotations

import logging
import pandas as pd
import numpy as np

from .metrics import compute_inertia


def initial_assignment(gdf, G, D: int, ideal_pop: float, pop_tolerance_ratio: float):
    """
    Builds districts one at a time using an inertia-guided growth algorithm.
    """
    logging.info("Step 3 of 5: Generating initial district map using Inertia-Guided Growth...")
    
    districts = [set() for _ in range(D)]
    unassigned_blocks = set(gdf["GEOID20"])
    min_pop = ideal_pop * (1 - pop_tolerance_ratio)
    
    # Pre-calculate data for performance
    block_pop_map = {row.GEOID20: row.P1_001N for row in gdf.itertuples()}
    
    for i in range(D - 1):
        logging.info(f"--- Building District {i+1}/{D} ---")
        
        # --- 1. Find the Seed Block (Cardinal Direction) ---
        # Filter GDF to only unassigned blocks
        remaining_gdf = gdf[gdf["GEOID20"].isin(unassigned_blocks)]
        # Sort to find the Northwest-most block of the remaining territory
        sorted_remaining = remaining_gdf.sort_values(by=["y", "x"], ascending=[False, True])
        
        if sorted_remaining.empty:
            logging.warning(f"No more blocks to assign, stopping at District {i}.")
            break
            
        seed_block = sorted_remaining.iloc[0]["GEOID20"]
        
        # --- 2. Grow the District ---
        current_district_blocks = {seed_block}
        unassigned_blocks.remove(seed_block)
        current_pop = block_pop_map[seed_block]

        while current_pop < min_pop:
            # Find all unassigned blocks adjacent to the current district
            frontier = {
                neighbor
                for block in current_district_blocks
                for neighbor in G.neighbors(block)
                if neighbor in unassigned_blocks
            }

            if not frontier:
                # This can happen if a district gets trapped by other completed districts
                logging.warning(f"District {i+1} ran out of neighbors before reaching target pop.")
                break

            # Find the best block on the frontier to add
            best_block_to_add = None
            lowest_next_inertia = float('inf')

            for frontier_block in frontier:
                # Create a hypothetical district to test the inertia
                hypothetical_district = current_district_blocks | {frontier_block}
                
                # Calculate the inertia if we add this block
                what_if_inertia = compute_inertia(hypothetical_district, gdf)
                
                if what_if_inertia < lowest_next_inertia:
                    lowest_next_inertia = what_if_inertia
                    best_block_to_add = frontier_block
            
            if best_block_to_add:
                current_district_blocks.add(best_block_to_add)
                unassigned_blocks.remove(best_block_to_add)
                current_pop += block_pop_map[best_block_to_add]
            else:
                # Should not happen if frontier is not empty
                logging.warning(f"Could not find a best block for District {i+1}, growth stalled.")
                break
        
        districts[i] = current_district_blocks
        logging.info(f"District {i+1} complete with population {current_pop:,}.")

    # --- 3. The Final District ---
    # The last district is whatever remains
    final_district_idx = D - 1
    districts[final_district_idx] = unassigned_blocks
    final_pop = sum(block_pop_map[b] for b in unassigned_blocks)
    logging.info(f"--- Building District {D}/{D} ---")
    logging.info(f"District {D} complete with remaining {len(unassigned_blocks)} blocks and population {final_pop:,}.")

    return [list(d) for d in districts]
