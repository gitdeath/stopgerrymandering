from __future__ import annotations

import logging
import math
import pandas as pd
import numpy as np


def initial_assignment(gdf, G, D: int, ideal_pop: float, pop_tolerance_ratio: float):
    """
    Builds districts using a prioritized, layer-by-layer BFS growth, seeded from
    8 equally spaced points on the state's exterior border.
    """
    logging.info("Step 3 of 5: Generating initial district map using Prioritized BFS Growth from Edges...")
    
    districts = [set() for _ in range(D)]
    unassigned_blocks = set(gdf["GEOID20"])
    
    # --- Pre-calculation ---
    geoid_idx = gdf.columns.get_loc("GEOID20")
    pop_idx = gdf.columns.get_loc("P1_001N")
    x_idx = gdf.columns.get_loc("x")
    y_idx = gdf.columns.get_loc("y")

    block_pop_map = {row[geoid_idx]: int(row[pop_idx]) for row in gdf.itertuples(index=False, name=None)}
    
    max_pop = ideal_pop * (1 + pop_tolerance_ratio)

    # --- 1. EDGE SEEDING PHASE ---
    logging.info(f"Seeding {D} districts from the state border...")

    # Find the exterior boundary of the entire state
    state_boundary = gdf.unary_union.exterior
    
    # Find all blocks that lie on the exterior boundary
    border_blocks_gdf = gdf[gdf.geometry.intersects(state_boundary)]
    
    if len(border_blocks_gdf) < D:
        raise RuntimeError("Not enough border blocks to seed all districts.")
        
    # Find the geometric center of the state to sort border blocks angularly
    state_centroid = gdf.unary_union.centroid
    cx, cy = state_centroid.x, state_centroid.y
    
    # Calculate the angle of each border block relative to the state's center
    border_blocks_gdf['angle'] = border_blocks_gdf.apply(
        lambda row: math.atan2(row['y'] - cy, row['x'] - cx),
        axis=1
    )
    
    # Sort the border blocks by angle to get a clockwise ordering
    sorted_border_blocks = border_blocks_gdf.sort_values('angle')

    # Select D equally spaced seeds from the sorted border blocks
    seed_indices = np.linspace(0, len(sorted_border_blocks) - 1, D, dtype=int)
    seeds = [sorted_border_blocks.iloc[i]["GEOID20"] for i in seed_indices]

    pop_per_district = np.zeros(D)
    
    # queues will hold the blocks for the current layer of each district
    queues = [[] for _ in range(D)]
    visited = set()

    for i, seed_block in enumerate(seeds):
        if seed_block not in unassigned_blocks:
            # Handle cases where a chosen seed was already taken by a nearby seed
            continue
            
        districts[i].add(seed_block)
        unassigned_blocks.remove(seed_block)
        visited.add(seed_block)
        
        pop_per_district[i] += block_pop_map[seed_block]
        queues[i] = [seed_block]
    
    # --- 2. PRIORITIZED LAYER-BY-LAYER GROWTH ---
    logging.info("Starting prioritized layer-by-layer growth...")
    districts_at_max_pop = [False] * D
    
    while unassigned_blocks:
        active_pops = [p if not districts_at_max_pop[i] else float('inf') for i, p in enumerate(pop_per_district)]
        
        if all(p == float('inf') for p in active_pops):
            logging.info("All districts are full. Moving to stalemate resolution.")
            break
            
        i = np.argmin(active_pops)

        if not queues[i]:
            districts_at_max_pop[i] = True
            continue

        frontier = set()
        for block in queues[i]:
            for neighbor in G.neighbors(block):
                if neighbor in unassigned_blocks and neighbor not in visited:
                    frontier.add(neighbor)
        
        if not frontier:
            districts_at_max_pop[i] = True
            continue

        # Mark frontier as visited now to prevent other districts from claiming it in the same pass
        visited.update(frontier)
        
        frontier_pop = sum(block_pop_map[b] for b in frontier)

        if pop_per_district[i] + frontier_pop > max_pop:
            districts_at_max_pop[i] = True
            # Since we can't add this layer, "un-visit" the frontier so other districts can claim it
            visited.difference_update(frontier)
            continue

        districts[i].update(frontier)
        unassigned_blocks.difference_update(frontier)
        pop_per_district[i] += frontier_pop
        
        queues[i] = list(frontier)

        blocks_assigned_total = len(visited)
        if blocks_assigned_total > 0 and blocks_assigned_total % 10000 == 0:
             logging.info(f"Assigned {blocks_assigned_total} / {len(gdf)} blocks...")
             
    # --- 3. STALEMATE RESOLUTION ---
    if unassigned_blocks:
        logging.warning(f"Assigning {len(unassigned_blocks)} remaining trapped blocks...")
        membership = {b: i for i, d in enumerate(districts) for b in d if d}
        for block in sorted(list(unassigned_blocks)):
            adj_districts = {membership.get(n) for n in G.neighbors(block) if membership.get(n) is not None}
            if adj_districts:
                best_neighbor_dist = min(adj_districts, key=lambda d_idx: pop_per_district[d_idx])
                districts[best_neighbor_dist].add(block)
                pop_per_district[best_neighbor_dist] += block_pop_map.get(block, 0)
            else:
                districts[np.argmin(pop_per_district)].add(block)
                pop_per_district[np.argmin(pop_per_district)] += block_pop_map.get(block, 0)

    logging.info("Initial assignment complete.")
    return [list(d) for d in districts]
