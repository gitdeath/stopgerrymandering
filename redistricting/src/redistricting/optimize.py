from __future__ import annotations

import logging
import random
from collections import Counter
import networkx as nx

from .metrics import objective, is_contiguous


def fix_contiguity(districts, gdf, G: nx.Graph):
    """
    Stage 1: Finds and repairs non-contiguous districts by moving entire islands
    and running a final smoothing pass to clean up small irregularities.
    """
    current_districts = [set(d) for d in districts]
    
    # --- Main Island-Fixing Loop ---
    while True:
        fixes_made_in_pass = 0
        membership = {block: i for i, dist in enumerate(current_districts) for block in dist}
        
        for i, district in enumerate(current_districts):
            if not district:
                continue

            components = list(nx.connected_components(G.subgraph(district)))
            if len(components) <= 1:
                continue

            components.sort(key=len, reverse=True)
            main_component = components[0]
            islands = components[1:]
            
            logging.warning(f"District {i+1} is not contiguous. Found {len(islands)} island(s). Fixing.")
            current_districts[i] = set(main_component)
            
            for island in islands:
                external_neighbors = nx.node_boundary(G, island)
                neighbor_districts = [membership.get(n) for n in external_neighbors if membership.get(n) is not None and membership.get(n) != i]

                if not neighbor_districts:
                    logging.error(f"FATAL: Island from D{i+1} has NO neighbors. Cannot fix.")
                    current_districts[i].update(island)
                    continue

                most_common_neighbor_dist = Counter(neighbor_districts).most_common(1)[0][0]
                current_districts[most_common_neighbor_dist].update(island)
                fixes_made_in_pass += len(island)

        if fixes_made_in_pass == 0:
            logging.info("Main contiguity pass complete. No more large islands found.")
            break
            
    # --- Final Smoothing/Cleanup Pass ---
    logging.info("Starting final contiguity cleanup pass...")
    smoothed_districts = [set(d) for d in current_districts]
    membership = {block: i for i, dist in enumerate(smoothed_districts) for block in dist}
    
    # Iterate over a sorted list for determinism
    all_blocks = sorted(list(G.nodes()))
    
    for block in all_blocks:
        current_district_idx = membership.get(block)
        if current_district_idx is None: continue

        neighbor_districts = [membership.get(n) for n in G.neighbors(block) if membership.get(n) is not None]
        if not neighbor_districts: continue

        most_common_neighbor = Counter(neighbor_districts).most_common(1)[0][0]

        if current_district_idx != most_common_neighbor:
            source_district = smoothed_districts[current_district_idx]
            if len(source_district) > 1 and not is_contiguous(source_district - {block}, G):
                continue 

            logging.warning(f"SMOOTHING: Reassigning rogue block {block} from D{current_district_idx+1} to majority neighbor D{most_common_neighbor+1}.")
            smoothed_districts[current_district_idx].remove(block)
            smoothed_districts[most_common_neighbor].add(block)
            membership[block] = most_common_neighbor
    
    # Final verification
    for i, d in enumerate(smoothed_districts):
        if d and not is_contiguous(d, G):
            raise RuntimeError(f"Contiguity fix failed for District {i+1} even after smoothing.")

    logging.info("Contiguity repair complete.")
    return [list(d) for d in smoothed_districts]


def powerful_balancer(districts, gdf, G, ideal_pop, pop_tolerance_ratio):
    """
    Stage 2: A powerful, resilient balancer that searches for a valid chunk to move
    without breaking district contiguity.
    """
    current = [set(d) for d in districts]
    min_pop = ideal_pop * (1 - pop_tolerance_ratio)
    max_pop = ideal_pop * (1 + pop_tolerance_ratio)
    
    MAX_BALANCER_ITERATIONS = 200
    for i in range(MAX_BALANCER_ITERATIONS):
        pop_per_district = [sum(G.nodes[b]["pop"] for b in d) for d in current]
        
        over_populated_districts = sorted(
            [(pop, idx) for idx, pop in enumerate(pop_per_district) if pop > max_pop],
            reverse=True
        )
        under_populated_districts = sorted(
            [(pop, idx) for idx, pop in enumerate(pop_per_district) if pop < min_pop]
        )

        if not over_populated_districts or not under_populated_districts:
            logging.info(f"Balancer Iteration {i+1}: No more over/under-populated districts to fix.")
            break

        source_pop, source_idx = over_populated_districts[0]
        target_pop, target_idx = under_populated_districts[0]

        pop_surplus = source_pop - max_pop
        pop_needed = min_pop - target_pop
        
        # Target a small chunk to avoid overshooting
        chunk_target_pop = min(pop_needed, pop_surplus, ideal_pop * 0.02)

        border_blocks = sorted(list({
            b for b in current[source_idx] 
            if any(n in current[target_idx] for n in G.neighbors(b))
        }))

        if not border_blocks:
            logging.error(f"Balancer cannot find a border between the most over-populated district (D{source_idx+1}) and the most under-populated (D{target_idx+1}). Halting.")
            break
        
        chunk_to_move = None
        
        # Iterate through border blocks to find a valid starting point for a chunk
        for start_node in border_blocks:
            # Grow a chunk using BFS
            chunk = set()
            chunk_pop = 0
            queue = [start_node]
            visited = {start_node}

            while queue:
                block = queue.pop(0)
                block_pop = G.nodes[block]["pop"]
                
                # Stop if chunk is big enough; don't add the current block
                if chunk_pop + block_pop > chunk_target_pop:
                    continue

                chunk.add(block)
                chunk_pop += block_pop
                
                for neighbor in G.neighbors(block):
                    if neighbor in current[source_idx] and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # After growing a potential chunk, verify it's safe to move
            if chunk:
                source_after_move = current[source_idx] - chunk
                target_after_move = current[target_idx] | chunk
                
                # Key validation: ensure both districts remain contiguous
                if is_contiguous(source_after_move, G) and is_contiguous(target_after_move, G):
                    chunk_to_move = chunk
                    break # Found a valid chunk, stop searching
        
        if chunk_to_move:
            moved_pop = sum(G.nodes[b]["pop"] for b in chunk_to_move)
            logging.info(f"Balancer: Moving {len(chunk_to_move)} blocks ({moved_pop:,} pop) from D{source_idx+1} to D{target_idx+1}.")
            current[source_idx] -= chunk_to_move
            current[target_idx] |= chunk_to_move
        else:
            logging.warning(f"Balancer searched the entire border between D{source_idx+1} and D{target_idx+1} but could not find a valid chunk to move. Stopping.")
            break
    else:
        logging.warning("Balancer reached iteration limit.")

    logging.info("Powerful balancing complete.")
    return [list(d) for d in current]


def perfect_map(districts, gdf, G, ideal_pop, pop_tolerance_ratio, compactness_threshold):
    """Stage 3: The final, meticulous optimization."""
    current = [set(d) for d in districts]
    current_score = objective(current, gdf, G, ideal_pop, pop_tolerance_ratio, compactness_threshold)
    logging.info(f"Final perfecting starting score: {current_score:.2f}")

    iteration = 0
    while True:
        iteration += 1
        best_move, best_delta = None, 0.0
        membership = {b: i for i, d in enumerate(current) for b in d}
        all_blocks = list(G.nodes)

        for block in all_blocks:
            from_idx = membership.get(block)
            if from_idx is None: continue
            neighbor_districts = {
                membership.get(n) for n in G.neighbors(block)
                if membership.get(n) is not None and membership[n] != from_idx
            }

            for to_idx in neighbor_districts:
                trial = [set(d) for d in current]
                trial[from_idx].remove(block)
                trial[to_idx].add(block)
                new_score = objective(trial, gdf, G, ideal_pop, pop_tolerance_ratio, compactness_threshold)
                delta = current_score - new_score
                if delta > best_delta:
                    best_delta, best_move = delta, (block, from_idx, to_idx)

        if best_move:
            b, fidx, tidx = best_move
            current[fidx].remove(b)
            current[tidx].add(b)
            current_score -= best_delta
            logging.info(f"Perfecting Iteration {iteration}: Applied best move; new score {current_score:.2f}")
        else:
            logging.info(f"Perfecting Iteration {iteration}: No further improvements found.")
            break

    return [list(d) for d in current], current_score
