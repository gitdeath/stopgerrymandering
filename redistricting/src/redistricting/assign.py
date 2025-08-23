from __future__ import annotations

import logging
import math
import pandas as pd
import numpy as np
import networkx as nx

def _recursively_bisect_population(gdf_to_split, num_districts: int):
    """
    Helper function to recursively split a GeoDataFrame into a target number of
    zones with roughly equal population.
    """
    if num_districts <= 1:
        return [gdf_to_split]
    
    # Determine split direction based on geographic shape
    min_x, min_y, max_x, max_y = gdf_to_split.total_bounds
    is_wider = (max_x - min_x) > (max_y - min_y)
    split_coord = 'x' if is_wider else 'y'
    
    # Sort blocks along the chosen axis
    sorted_gdf = gdf_to_split.sort_values(split_coord)
    
    # Find the split point that divides the population in half
    total_pop = sorted_gdf['P1_001N'].sum()
    target_pop_half = total_pop / 2
    
    cumulative_pop = 0
    split_index = -1
    for index, row in sorted_gdf.iterrows():
        cumulative_pop += row['P1_001N']
        if cumulative_pop >= target_pop_half:
            split_index = index
            break
    
    # Perform the split
    if split_index == -1: # Handle empty or single-block zones
        return [gdf_to_split]

    split_loc = sorted_gdf.index.get_loc(split_index)
    part1_gdf = sorted_gdf.iloc[:split_loc]
    part2_gdf = sorted_gdf.iloc[split_loc:]
    
    # Recursively call on the two halves
    num_districts_1 = round(num_districts / 2)
    num_districts_2 = num_districts - num_districts_1
    
    zones = []
    if not part1_gdf.empty and num_districts_1 > 0:
        zones.extend(_recursively_bisect_population(part1_gdf, num_districts_1))
    if not part2_gdf.empty and num_districts_2 > 0:
        zones.extend(_recursively_bisect_population(part2_gdf, num_districts_2))
        
    return zones


def initial_assignment(gdf, G, D: int, ideal_pop: float, pop_tolerance_ratio: float):
    """
    Builds districts by first using recursive bisection to create zones, then
    seeding from the center of each zone's state border segment, and finally
    growing districts using a prioritized, layer-by-layer BFS.
    """
    logging.info("Step 3 of 5: Generating initial district map using Bisection Border Seeding...")
    
    districts = [set() for _ in range(D)]
    unassigned_blocks = set(gdf["GEOID20"])
    
    # Pre-calculation
    block_pop_map = {row['GEOID20']: int(row['P1_001N']) for _, row in gdf.iterrows()}
    max_pop = ideal_pop * (1 + pop_tolerance_ratio)

    # --- 1. RECURSIVE BISECTION & BORDER SEEDING ---
    logging.info(f"Bisecting state into {D} zones to find border seeds...")
    
    temp_gdf = gdf[['GEOID20', 'geometry', 'x', 'y']].copy()
    temp_gdf['P1_001N'] = temp_gdf['GEOID20'].map(block_pop_map)
    zones = _recursively_bisect_population(temp_gdf, D)

    logging.info("Finding seeds from the center of each zone's state border...")
    seeds = []
    state_boundary = gdf.unary_union.exterior

    for i, zone_gdf in enumerate(zones):
        if zone_gdf.empty: continue

        seed_point = None
        # Special rule for the last district if D is odd
        if D % 2 != 0 and i == D - 1:
            logging.info(f"Odd number of districts detected. Placing final seed at state's population center.")
            state_pop = gdf['P1_001N'].sum()
            s_centroid_x = (gdf['x'] * gdf['P1_001N']).sum() / state_pop
            s_centroid_y = (gdf['y'] * gdf['P1_001N']).sum() / state_pop
            seed_point = (s_centroid_x, s_centroid_y)
        else:
            # Find the part of the zone's border that is on the state's exterior
            zone_border_segment = zone_gdf.unary_union.boundary.intersection(state_boundary)
            if zone_border_segment.is_empty:
                # If a zone has no exterior border, use its population centroid as a fallback
                logging.warning(f"Zone {i+1} has no exterior border. Seeding from its population center instead.")
                zone_pop = zone_gdf['P1_001N'].sum()
                if zone_pop > 0:
                    centroid_x = (zone_gdf['x'] * zone_gdf['P1_001N']).sum() / zone_pop
                    centroid_y = (zone_gdf['y'] * zone_gdf['P1_001N']).sum() / zone_pop
                    seed_point = (centroid_x, centroid_y)
            else:
                # Find the center point of that border segment
                border_centroid = zone_border_segment.centroid
                seed_point = (border_centroid.x, border_centroid.y)

        if seed_point:
            # Find the actual block in the zone closest to the calculated seed point
            min_dist = float('inf')
            best_seed = None
            for _, block in zone_gdf.iterrows():
                dist_sq = (block['x'] - seed_point[0])**2 + (block['y'] - seed_point[1])**2
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    best_seed = block['GEOID20']
            if best_seed:
                seeds.append(best_seed)

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
    logging.info("Starting prioritized layer-by-layer growth from border seeds...")
    districts_at_max_pop = [False] * D
    
    while unassigned_blocks:
        active_pops = [p if not districts_at_max_pop[i] else float('inf') for i, p in enumerate(pop_per_district)]
        
        if all(p == float('inf') for p in active_pops):
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
        frontier_pop = sum(block_pop_map.get(b, 0) for b in frontier)

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
             
    # --- 3. FINAL CONSOLIDATION FOR LEFTOVERS ---
    if unassigned_blocks:
        logging.warning(f"Consolidating {len(unassigned_blocks)} remaining trapped blocks...")
        unassigned_subgraph = G.subgraph(unassigned_blocks)
        islands = list(nx.connected_components(unassigned_subgraph))
        membership = {b: i for i, d in enumerate(districts) for b in d if d}

        for island in islands:
            island_pop = sum(block_pop_map.get(b, 0) for b in island)
            external_neighbors = nx.node_boundary(G, island)
            neighbor_districts = {membership.get(n) for n in external_neighbors if membership.get(n) is not None}

            if not neighbor_districts:
                smallest_dist_idx = np.argmin(pop_per_district)
                districts[smallest_dist_idx].update(island)
                pop_per_district[smallest_dist_idx] += island_pop
                continue

            valid_neighbors = {d_idx for d_idx in neighbor_districts if pop_per_district[d_idx] + island_pop <= max_pop}
            
            if valid_neighbors:
                best_neighbor = min(valid_neighbors, key=lambda d_idx: pop_per_district[d_idx])
            else:
                best_neighbor = min(neighbor_districts, key=lambda d_idx: pop_per_district[d_idx] + island_pop)
                
            districts[best_neighbor].update(island)
            pop_per_district[best_neighbor] += island_pop

    logging.info("Initial assignment complete.")
    return [list(d) for d in districts if d]
