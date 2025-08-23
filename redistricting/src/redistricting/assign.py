from __future__ import annotations

import logging
import math
import networkx as nx
import pandas as pd
import numpy as np


def initial_assignment(gdf, G, D: int, ideal_pop: float, pop_tolerance_ratio: float):
    """
    Builds districts using a prioritized, layer-by-layer BFS growth, seeded from
    8 equally spaced points on the state's exterior border. Ends with an
    intelligent consolidation phase for any leftover blocks.
    """
    logging.info("Step 3 of 5: Generating initial district map using Prioritized BFS Growth from Edges...")
    
    districts = [set() for _ in range(D)]
    unassigned_blocks = set(gdf["GEOID20"])
    
    # --- Pre-calculation ---
    block_pop_map = {row['GEOID20']: int(row['P1_001N']) for _, row in gdf.iterrows()}
    
    max_pop = ideal_pop * (1 + pop_tolerance_ratio)

    # --- 1. EDGE SEEDING PHASE ---
    logging.info(f"Seeding {D} districts from the state border...")
    state_boundary = gdf.unary_union.exterior
    border_blocks_gdf = gdf[gdf.geometry.intersects(state_boundary)].copy()
    
    if len(border_blocks_gdf) < D:
        raise RuntimeError("Not enough border blocks to seed all districts.")
        
    state_centroid = gdf.unary_union.centroid
    cx, cy = state_centroid.x, state_centroid.y
    
    border_blocks_gdf['angle'] = border_blocks_gdf.apply(
        lambda row: math.atan2(row['y'] - cy, row['x'] - cx),
        axis=1
    )
    
    sorted_border_blocks = border_blocks_gdf.sort_values('angle')
    seed_indices = np.linspace(0, len(sorted_border_blocks) - 1, D, dtype=int)
    seeds = [sorted_border_blocks.iloc[i]["GEOID20"] for i in seed_indices]

    pop_per_district = np.zeros(D)
    queues = [[] for _ in range(D)]
    visited = set()

    for i, seed_block in enumerate(seeds):
        if seed_block not in unassigned_blocks: continue
            
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
            logging.info("All districts are full. Moving to consolidation phase.")
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

        visited.update(frontier)
        frontier_pop = sum(block_pop_map[b] for b in frontier)

        if pop_per_district[i] + frontier_pop > max_pop:
            districts_at_max_pop[i] = True
            visited.difference_update(frontier)
            continue

        districts[i].update(frontier)
        unassigned_blocks.difference_update(frontier)
        pop_per_district[i] += frontier_pop
        queues[i] = list(frontier)

        blocks_assigned_total = len(visited)
        if blocks_assigned_total > 0 and blocks_assigned_total % 10000 == 0:
             logging.info(f"Assigned {blocks_assigned_total} / {len(gdf)} blocks...")
             
    # --- 3. FINAL CONSOLIDATION (NEW STALEMATE LOGIC) ---
    if unassigned_blocks:
        logging.warning(f"Consolidating {len(unassigned_blocks)} remaining trapped blocks...")
        
        # Group leftover blocks into contiguous islands
        unassigned_subgraph = G.subgraph(unassigned_blocks)
        islands = list(nx.connected_components(unassigned_subgraph))
        
        membership = {b: i for i, d in enumerate(districts) for b in d if d}

        for island in islands:
            island_pop = sum(block_pop_map[b] for b in island)
            
            # Find all neighboring districts
            external_neighbors = nx.node_boundary(G, island)
            neighbor_districts = {membership.get(n) for n in external_neighbors if membership.get(n) is not None}

            if not neighbor_districts:
                logging.error(f"Could not find a neighbor for a trapped island of {len(island)} blocks. Assigning to smallest district overall.")
                smallest_dist_idx = np.argmin(pop_per_district)
                districts[smallest_dist_idx].update(island)
                pop_per_district[smallest_dist_idx] += island_pop
                continue

            # Primary rule: find neighbors that can accept the island without going over the population cap
            valid_neighbors = {
                d_idx for d_idx in neighbor_districts 
                if pop_per_district[d_idx] + island_pop <= max_pop
            }
            
            if valid_neighbors:
                # Give the island to the "hungriest" valid neighbor (furthest below ideal)
                best_neighbor = min(valid_neighbors, key=lambda d_idx: pop_per_district[d_idx])
                districts[best_neighbor].update(island)
                pop_per_district[best_neighbor] += island_pop
            else:
                # Fallback rule: if all neighbors are full, give it to the one it overpopulates the least
                best_neighbor = min(neighbor_districts, key=lambda d_idx: pop_per_district[d_idx] + island_pop)
                districts[best_neighbor].update(island)
                pop_per_district[best_neighbor] += island_pop

    logging.info("Initial assignment complete.")
    return [list(d) for d in districts if d]
