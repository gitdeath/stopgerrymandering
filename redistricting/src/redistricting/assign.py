from __future__ import annotations

import logging
import random
import pandas as pd
import numpy as np


def initial_assignment(gdf, G, D: int, ideal_pop: float, pop_tolerance_ratio: float):
    """
    Builds districts using a hybrid growth model:
    1. Non-competitive growth to establish stable, contiguous cores.
    2. Competitive growth to assign the remaining blocks between cores.
    """
    logging.info("Step 3 of 5: Generating initial district map using Hybrid Growth...")
    
    districts = [set() for _ in range(D)]
    unassigned_blocks = set(gdf["GEOID20"])
    total_blocks = len(unassigned_blocks)
    
    # --- Pre-calculation ---
    geoid_idx = gdf.columns.get_loc("GEOID20")
    pop_idx = gdf.columns.get_loc("P1_001N")
    x_idx = gdf.columns.get_loc("x")
    y_idx = gdf.columns.get_loc("y")

    block_pop_map = {row[geoid_idx]: int(row[pop_idx]) for row in gdf.itertuples(index=False, name=None)}
    block_coords_map = {row[geoid_idx]: (row[x_idx], row[y_idx]) for row in gdf.itertuples(index=False, name=None)}
    
    # --- 1. SEEDING PHASE ---
    logging.info(f"Seeding {D} districts...")
    populated_gdf = gdf[gdf["P1_001N"] > 0]
    sorted_populated = populated_gdf.sort_values(by=["y", "x"], ascending=[False, True])
    
    if len(sorted_populated) < D:
        raise ValueError("Not enough populated blocks to seed all districts.")
        
    seed_indices = np.linspace(0, len(sorted_populated) - 1, D, dtype=int)
    seeds = [sorted_populated.iloc[i]["GEOID20"] for i in seed_indices]

    pop_per_district = np.zeros(D)
    pop_x_per_district = np.zeros(D)
    pop_y_per_district = np.zeros(D)
    frontiers = [set() for _ in range(D)]

    for i, seed_block in enumerate(seeds):
        districts[i].add(seed_block)
        unassigned_blocks.remove(seed_block)
        
        pop_to_add = block_pop_map[seed_block]
        coords_to_add = block_coords_map[seed_block]

        pop_per_district[i] += pop_to_add
        pop_x_per_district[i] += coords_to_add[0] * pop_to_add
        pop_y_per_district[i] += coords_to_add[1] * pop_to_add
        
        for neighbor in G.neighbors(seed_block):
            if neighbor in unassigned_blocks:
                frontiers[i].add(neighbor)

    # --- 2A. PHASE A: NON-COMPETITIVE CORE GROWTH ---
    logging.info("Starting Phase A: Non-competitive core growth...")
    core_pop_target = ideal_pop * 0.40
    districts_at_core_pop = [False] * D
    
    while not all(districts_at_core_pop):
        blocks_assigned_in_round = 0
        
        # Build a set of all blocks on any frontier for efficient checking
        all_frontier_blocks = set.union(*frontiers)
        
        for i in range(D):
            if districts_at_core_pop[i] or not frontiers[i]:
                continue
            
            # Identify the uncontested frontier for this district
            other_frontiers = set.union(*(f for j, f in enumerate(frontiers) if i != j))
            uncontested_frontier = frontiers[i] - other_frontiers

            if not uncontested_frontier:
                continue

            current_pop = pop_per_district[i]
            if current_pop == 0: continue
            
            centroid_x = pop_x_per_district[i] / current_pop
            centroid_y = pop_y_per_district[i] / current_pop
            
            best_block_to_add = None
            lowest_inertia_gain = float('inf')

            for frontier_block in uncontested_frontier:
                pop = block_pop_map[frontier_block]
                coords = block_coords_map[frontier_block]
                dist_sq = (coords[0] - centroid_x)**2 + (coords[1] - centroid_y)**2
                inertia_gain = pop * dist_sq
                
                if inertia_gain < lowest_inertia_gain:
                    lowest_inertia_gain = inertia_gain
                    best_block_to_add = frontier_block

            if best_block_to_add:
                pop_to_add = block_pop_map[best_block_to_add]
                coords_to_add = block_coords_map[best_block_to_add]
                
                districts[i].add(best_block_to_add)
                pop_per_district[i] += pop_to_add
                pop_x_per_district[i] += coords_to_add[0] * pop_to_add
                pop_y_per_district[i] += coords_to_add[1] * pop_to_add
                
                unassigned_blocks.remove(best_block_to_add)
                
                for f in frontiers:
                    f.discard(best_block_to_add)

                for neighbor in G.neighbors(best_block_to_add):
                    if neighbor in unassigned_blocks:
                        frontiers[i].add(neighbor)

                blocks_assigned_in_round += 1

                if pop_per_district[i] >= core_pop_target:
                    districts_at_core_pop[i] = True
                    logging.info(f"District {i+1} has reached its core population target.")
        
        if blocks_assigned_in_round == 0:
            logging.warning("Non-competitive growth stalled. Moving to competitive phase.")
            break

    logging.info("Phase A complete. All districts have formed stable cores.")
    
    # --- 2B. PHASE B: COMPETITIVE INFILLING ---
    logging.info("Starting Phase B: Competitive infilling...")
    max_pop = ideal_pop * (1 + pop_tolerance_ratio)
    districts_at_max_pop = [False] * D
    MAX_FRONTIER_SAMPLE = 500
    round_num = 0

    while unassigned_blocks:
        round_num += 1
        blocks_assigned_in_round = 0
        
        for i in range(D):
            if districts_at_max_pop[i] or not frontiers[i]:
                continue

            current_pop = pop_per_district[i]
            if current_pop == 0: continue
            
            centroid_x = pop_x_per_district[i] / current_pop
            centroid_y = pop_y_per_district[i] / current_pop
            
            best_block_to_add = None
            lowest_inertia_gain = float('inf')
            
            frontier_to_check = frontiers[i]
            if len(frontier_to_check) > MAX_FRONTIER_SAMPLE:
                sorted_frontier = sorted(
                    frontier_to_check,
                    key=lambda b: (block_coords_map[b][0] - centroid_x)**2 + (block_coords_map[b][1] - centroid_y)**2
                )
                frontier_to_check = sorted_frontier[:MAX_FRONTIER_SAMPLE]

            for frontier_block in frontier_to_check:
                pop = block_pop_map[frontier_block]
                coords = block_coords_map[frontier_block]
                dist_sq = (coords[0] - centroid_x)**2 + (coords[1] - centroid_y)**2
                inertia_gain = pop * dist_sq
                
                if inertia_gain < lowest_inertia_gain:
                    lowest_inertia_gain = inertia_gain
                    best_block_to_add = frontier_block
            
            if best_block_to_add:
                pop_to_add = block_pop_map[best_block_to_add]
                coords_to_add = block_coords_map[best_block_to_add]
                
                districts[i].add(best_block_to_add)
                pop_per_district[i] += pop_to_add
                pop_x_per_district[i] += coords_to_add[0] * pop_to_add
                pop_y_per_district[i] += coords_to_add[1] * pop_to_add
                
                unassigned_blocks.remove(best_block_to_add)
                
                for f in frontiers:
                    f.discard(best_block_to_add)

                for neighbor in G.neighbors(best_block_to_add):
                    if neighbor in unassigned_blocks:
                        frontiers[i].add(neighbor)

                blocks_assigned_in_round += 1

                if pop_per_district[i] >= max_pop:
                    districts_at_max_pop[i] = True
                    logging.info(f"District {i+1} has reached its population cap.")
        
        blocks_assigned_total = total_blocks - len(unassigned_blocks)
        if round_num % 20 == 0: # Log progress less frequently in this phase
             logging.info(f"Assigned {blocks_assigned_total} / {total_blocks} blocks...")

        if blocks_assigned_in_round == 0 and unassigned_blocks:
            logging.warning(f"Stalemate detected with {len(unassigned_blocks)} trapped blocks. Assigning them.")
            membership = {b: i for i, d in enumerate(districts) for b in d}
            for block in sorted(list(unassigned_blocks)):
                adj_districts = {membership[n] for n in G.neighbors(block) if n in membership}
                if adj_districts:
                    best_neighbor_dist = min(adj_districts, key=lambda d_idx: (pop_per_district[d_idx], d_idx))
                    districts[best_neighbor_dist].add(block)
                else:
                    districts[np.argmin(pop_per_district)].add(block)
            unassigned_blocks.clear()
            break
             
    logging.info("Initial assignment complete.")
    return [list(d) for d in districts]
