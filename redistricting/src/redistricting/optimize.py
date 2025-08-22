from __future__ import annotations

import logging
import random
from collections import Counter
import networkx as nx

from .metrics import objective, is_contiguous


def fix_contiguity(districts, gdf, G: nx.Graph):
    """
    Stage 1: Finds and repairs non-contiguous districts by moving entire islands.
    """
    current_districts = [set(d) for d in districts]
    
    while True:
        fixes_made_in_pass = 0
        
        # Rebuild membership map at the start of each full pass
        membership = {block: i for i, dist in enumerate(current_districts) for block in dist}
        
        for i, district in enumerate(current_districts):
            if not district:
                continue

            components = list(nx.connected_components(G.subgraph(district)))
            if len(components) <= 1:
                continue

            # Identify the main component and the islands
            components.sort(key=len, reverse=True)
            main_component = components[0]
            islands = components[1:]
            
            logging.warning(f"District {i+1} is not contiguous. Found {len(islands)} island(s).")

            # Update the district to only contain the main component for now
            current_districts[i] = set(main_component)
            
            for island in islands:
                # Find all external neighbors of the entire island
                external_neighbors = nx.node_boundary(G, island)
                neighbor_districts = [membership.get(n) for n in external_neighbors if membership.get(n) is not None and membership.get(n) != i]

                if not neighbor_districts:
                    logging.error(f"FATAL: Island of {len(island)} blocks from D{i+1} has NO neighbors. Cannot fix.")
                    # In a real scenario, you might assign to the least populated district or handle this case
                    # For now, we'll add it back to its original district to avoid losing blocks
                    current_districts[i].update(island)
                    continue

                # Find the most common neighboring district to donate the island to
                most_common_neighbor_dist = Counter(neighbor_districts).most_common(1)[0][0]
                
                logging.warning(
                    f"CONTIGUITY FIX: Moving an island of {len(island)} blocks from District {i+1} "
                    f"to its most common neighbor, District {most_common_neighbor_dist+1}."
                )
                
                # Move the entire island at once
                current_districts[most_common_neighbor_dist].update(island)
                fixes_made_in_pass += len(island)

        # If a full pass over all districts results in no fixes, we are done.
        if fixes_made_in_pass == 0:
            break
            
    # Final verification
    for i, d in enumerate(current_districts):
        if d and not is_contiguous(d, G):
            raise RuntimeError(f"Contiguity fix failed for District {i+1} after all passes.")

    logging.info("Contiguity repair complete.")
    return [list(d) for d in current_districts]


def powerful_balancer(districts, gdf, G, ideal_pop, pop_tolerance_ratio):
    """Stage 2: A powerful balancer that moves contiguous chunks of population."""
    current = [set(d) for d in districts]
    min_pop = ideal_pop * (1 - pop_tolerance_ratio)
    max_pop = ideal_pop * (1 + pop_tolerance_ratio)
    
    for i in range(200): # Limit iterations to prevent infinite loops
        pop_per_district = [sum(G.nodes[b]["pop"] for b in d) for d in current]
        
        over_populated_districts = [
            (pop, idx) for idx, pop in enumerate(pop_per_district) if pop > max_pop
        ]
        under_populated_districts = [
            (pop, idx) for idx, pop in enumerate(pop_per_district) if pop < min_pop
        ]

        if not over_populated_districts or not under_populated_districts:
            logging.info(f"Balancer Iteration {i+1}: No more over/under-populated districts to fix.")
            break

        over_populated_districts.sort(reverse=True)
        under_populated_districts.sort()

        source_idx = over_populated_districts[0][1]
        target_idx = under_populated_districts[0][1]

        pop_needed = min_pop - pop_per_district[target_idx]
        pop_surplus = pop_per_district[source_idx] - max_pop
        
        chunk_target_pop = min(pop_needed, pop_surplus, ideal_pop * 0.02)

        border_blocks = {
            b for b in current[source_idx] 
            if any(n in current[target_idx] for n in G.neighbors(b))
        }
        
        if not border_blocks:
            logging.warning(f"Balancer could not find border between D{source_idx+1} and D{target_idx+1}. Stopping.")
            break

        # Start BFS from a deterministic, random sample of border blocks
        start_nodes = sorted(list(border_blocks))
        random.Random(i).shuffle(start_nodes)
        
        chunk = set()
        chunk_pop = 0
        queue = start_nodes[:5] # Start search from a few points on the border
        visited = set(queue)

        while queue:
            block = queue.pop(0)
            block_pop = G.nodes[block]["pop"]
            
            if chunk_pop + block_pop > chunk_target_pop:
                continue

            chunk.add(block)
            chunk_pop += block_pop
            
            for neighbor in G.neighbors(block):
                if neighbor in current[source_idx] and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        if chunk:
            if is_contiguous(current[source_idx] - chunk, G):
                logging.info(f"Balancer: Moving {len(chunk)} blocks ({chunk_pop:,} pop) "
                             f"from D{source_idx+1} to D{target_idx+1}.")
                current[source_idx] -= chunk
                current[target_idx] |= chunk
            else:
                logging.warning(f"Balancer found a chunk to move but it would break contiguity. Stopping.")
                break
        else:
            logging.info("Balancer could not find a suitable chunk to move. Stopping.")
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
