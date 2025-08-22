from __future__ import annotations

import logging
import pandas as pd
import numpy as np


def initial_assignment(gdf, G, D: int, ideal_pop: float, pop_tolerance_ratio: float):
    """
    Builds districts using a prioritized, layer-by-layer BFS growth. At each step,
    the district with the smallest population adds its entire next 'ring' of blocks,
    ensuring balanced, organic growth.
    """
    logging.info("Step 3 of 5: Generating initial district map using Prioritized BFS Growth...")
    
    districts = [set() for _ in range(D)]
    unassigned_blocks = set(gdf["GEOID20"])
    
    # --- Pre-calculation ---
    geoid_idx = gdf.columns.get_loc("GEOID20")
    pop_idx = gdf.columns.get_loc("P1_001N")
    x_idx = gdf.columns.get_loc("x")
    y_idx = gdf.columns.get_loc("y")

    block_pop_map = {row[geoid_idx]: int(row[pop_idx]) for row in gdf.itertuples(index=False, name=None)}
    
    max_pop = ideal_pop * (1 + pop_tolerance_ratio)

    # --- 1. SEEDING PHASE ---
    logging.info(f"Seeding {D} districts...")
    
    # --- FIX: Simplify and correct the coordinate sorting ---
    coords_df = gdf[['GEOID20', 'y', 'x']].set_index('GEOID20')
    coords_df = coords_df[coords_df.index.isin(unassigned_blocks)]
    sorted_coords = coords_df.sort_values(by=["y", "x"], ascending=[False, True])
    # --- END FIX ---
    
    if len(sorted_coords) < D:
        raise ValueError("Not enough blocks to seed all districts.")
        
    seed_indices = np.linspace(0, len(sorted_coords) - 1, D, dtype=int)
    seeds = [sorted_coords.index[i] for i in seed_indices]

    pop_per_district = np.zeros(D)
    
    # queues will hold the blocks for the current layer of each district
    queues = [[] for _ in range(D)]
    visited = set()

    for i, seed_block in enumerate(seeds):
        districts[i].add(seed_block)
        unassigned_blocks.remove(seed_block)
        visited.add(seed_block)
        
        pop_per_district[i] += block_pop_map[seed_block]
        queues[i] = [seed_block]
    
    # --- 2. PRIORITIZED LAYER-BY-LAYER GROWTH ---
    logging.info("Starting prioritized layer-by-layer growth...")
    districts_at_max_pop = [False] * D
    
    while unassigned_blocks:
        # Find the district with the lowest current population that is not yet full
        active_pops = [p if not districts_at_max_pop[i] else float('inf') for i, p in enumerate(pop_per_district)]
        
        if all(p == float('inf') for p in active_pops):
            logging.info("All districts are full. Moving to stalemate resolution.")
            break
            
        i = np.argmin(active_pops) # This is the index of the smallest district

        if not queues[i]:
            # This district has no more blocks to expand from, so it is done.
            districts_at_max_pop[i] = True
            continue

        # Get the next layer (frontier) of blocks for the chosen district
        frontier = set()
        for block in queues[i]:
            for neighbor in G.neighbors(block):
                if neighbor in unassigned_blocks and neighbor not in visited:
                    frontier.add(neighbor)
        
        if not frontier:
            # This district is walled off, so it is done.
            districts_at_max_pop[i] = True
            continue

        frontier_pop = sum(block_pop_map[b] for b in frontier)

        # Check if adding the entire layer would overpopulate the district
        if pop_per_district[i] + frontier_pop > max_pop:
            districts_at_max_pop[i] = True
            continue # Skip this district's turn; it can't take its next full layer

        # If safe, assign the entire layer
        districts[i].update(frontier)
        unassigned_blocks.difference_update(frontier)
        visited.update(frontier)
        pop_per_district[i] += frontier_pop
        
        # The new layer becomes the queue for the next turn
        queues[i] = list(frontier)

        blocks_assigned_total = len(visited)
        if blocks_assigned_total > 0 and blocks_assigned_total % 10000 == 0:
             logging.info(f"Assigned {blocks_assigned_total} / {len(gdf)} blocks...")
             
    # --- 3. STALEMATE RESOLUTION ---
    if unassigned_blocks:
        logging.warning(f"Assigning {len(unassigned_blocks)} remaining trapped blocks...")
        membership = {b: i for i, d in enumerate(districts) for b in d}
        for block in sorted(list(unassigned_blocks)):
            adj_districts = {membership.get(n) for n in G.neighbors(block) if membership.get(n) is not None}
            if adj_districts:
                best_neighbor_dist = min(adj_districts, key=lambda d_idx: pop_per_district[d_idx])
                districts[best_neighbor_dist].add(block)
                pop_per_district[best_neighbor_dist] += block_pop_map.get(block, 0)
            else: # Should be rare, but handle isolated blocks
                districts[np.argmin(pop_per_district)].add(block)
                pop_per_district[np.argmin(pop_per_district)] += block_pop_map.get(block, 0)

    logging.info("Initial assignment complete.")
    return [list(d) for d in districts]
