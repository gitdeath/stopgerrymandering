from __future__ import annotations

import logging
import random
from collections import Counter
import networkx as nx

from .metrics import objective, is_contiguous, compute_inertia


def fix_contiguity(districts, gdf, G: nx.Graph):
    # This function is now correct and does not need changes.
    current_districts = [set(d) for d in districts]
    
    fixes_made = True
    while fixes_made:
        fixes_made = False
        pop_per_district = [sum(G.nodes[b]["pop"] for b in d) for d in current_districts]
        membership = {block: i for i, dist in enumerate(current_districts) for block in dist}
        
        for i, district in enumerate(current_districts):
            if not district: continue

            components = list(nx.connected_components(G.subgraph(district)))
            if len(components) <= 1: continue

            fixes_made = True
            components.sort(key=len, reverse=True)
            main_component = components[0]
            islands = components[1:]
            
            logging.warning(f"District {i+1} is not contiguous. Found {len(islands)} island(s). Fixing.")
            current_districts[i] = set(main_component)
            
            for island in islands:
                external_neighbors = nx.node_boundary(G, island)
                neighbor_districts = {membership.get(n) for n in external_neighbors if membership.get(n) is not None and membership.get(n) != i}

                if not neighbor_districts:
                    logging.error(f"FATAL: Island from D{i+1} has NO neighbors. Cannot fix.")
                    current_districts[i].update(island)
                    continue

                best_neighbor_dist = min(neighbor_districts, key=lambda d_idx: pop_per_district[d_idx])
                current_districts[best_neighbor_dist].update(island)
    
    for i, d in enumerate(current_districts):
        if d and not is_contiguous(d, G):
            raise RuntimeError(f"Contiguity fix failed for District {i+1}.")

    logging.info("Contiguity repair complete.")
    return [list(d) for d in current_districts]


def powerful_balancer(districts, gdf, G, ideal_pop, pop_tolerance_ratio):
    # This function is now correct and does not need changes.
    current = [set(d) for d in districts]
    min_pop = ideal_pop * (1 - pop_tolerance_ratio)
    max_pop = ideal_pop * (1 + pop_tolerance_ratio)
    
    MAX_BALANCER_ITERATIONS = 500
    for i in range(MAX_BALANCER_ITERATIONS):
        pop_per_district = [sum(G.nodes[b]["pop"] for b in d) for d in current]
        
        over_populated_districts = [idx for idx, pop in enumerate(pop_per_district) if pop > max_pop]
        under_populated_districts = [idx for idx, pop in enumerate(pop_per_district) if pop < min_pop]

        if not over_populated_districts or not under_populated_districts:
            logging.info(f"Balancer Iteration {i+1}: Populations are balanced.")
            break
        
        move_made_this_iteration = False
        for source_idx in over_populated_districts:
            for target_idx in under_populated_districts:
                border_blocks = sorted(list({b for b in current[source_idx] if any(n in current[target_idx] for n in G.neighbors(b))}))

                if not border_blocks: continue

                pop_surplus = pop_per_district[source_idx] - max_pop
                pop_needed = min_pop - pop_per_district[target_idx]
                chunk_target_pop = min(pop_needed, pop_surplus, ideal_pop * 0.02)
                
                chunk_to_move = None
                for start_node in border_blocks:
                    chunk = set()
                    chunk_pop = 0
                    queue = [start_node]
                    visited = {start_node}

                    while queue:
                        block = queue.pop(0)
                        block_pop = G.nodes[block]["pop"]
                        if chunk_pop + block_pop > chunk_target_pop: continue
                        chunk.add(block)
                        chunk_pop += block_pop
                        for neighbor in G.neighbors(block):
                            if neighbor in current[source_idx] and neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)
                    
                    if chunk:
                        source_after_move = current[source_idx] - chunk
                        target_after_move = current[target_idx] | chunk
                        if is_contiguous(source_after_move, G) and is_contiguous(target_after_move, G):
                            chunk_to_move = chunk
                            break
                
                if chunk_to_move:
                    moved_pop = sum(G.nodes[b]["pop"] for b in chunk_to_move)
                    logging.info(f"Balancer: Moving {len(chunk_to_move)} blocks ({moved_pop:,} pop) from D{source_idx+1} to D{target_idx+1}.")
                    current[source_idx] -= chunk_to_move
                    current[target_idx] |= chunk_to_move
                    move_made_this_iteration = True
                    break
            
            if move_made_this_iteration:
                break
        
        if not move_made_this_iteration:
            logging.info(f"Balancer Iteration {i+1}: Searched all pairs but found no valid moves.")
            break
    else:
        logging.warning("Balancer reached iteration limit.")

    logging.info("Powerful balancing complete.")
    return [list(d) for d in current]


def perfect_map(districts, gdf, G, ideal_pop, pop_tolerance_ratio):
    """
    Stage 3: The final, meticulous optimization using the hybrid score.
    """
    logging.info("Starting Stage 3: Final Perfecting...")
    current = [set(d) for d in districts]
    current_score = objective(current, gdf, G, ideal_pop, pop_tolerance_ratio)
    logging.info(f"Final perfecting starting score: {current_score:.2f}")

    MAX_PERFECTING_ITERATIONS = 50
    for iteration in range(1, MAX_PERFECTING_ITERATIONS + 1):
        best_move, best_delta = None, 0.0
        membership = {b: i for i, d in enumerate(current) for b in d}
        
        all_border_blocks = list({
            b for i, d in enumerate(current) for b in d 
            if any(membership.get(n) is not None and membership.get(n) != i for n in G.neighbors(b))
        })

        # --- THE PERFORMANCE FIX ---
        # Instead of checking all border blocks, check a more reasonably sized random sample.
        SAMPLE_SIZE = 500
        if len(all_border_blocks) > SAMPLE_SIZE:
            blocks_to_check = random.sample(all_border_blocks, SAMPLE_SIZE)
        else:
            blocks_to_check = all_border_blocks
        
        logging.debug(f"Perfecting Iteration {iteration}: Checking {len(blocks_to_check)} border blocks for improvements...")

        for block in blocks_to_check:
            from_idx = membership.get(block)
            if from_idx is None: continue
            
            neighbor_districts = {
                membership[n] for n in G.neighbors(block) if membership.get(n) is not None and membership[n] != from_idx
            }

            for to_idx in neighbor_districts:
                trial = [set(d) for d in current]
                trial[from_idx].remove(block)
                trial[to_idx].add(block)
                
                if not is_contiguous(trial[from_idx], G): continue
                
                new_score = objective(trial, gdf, G, ideal_pop, pop_tolerance_ratio)
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
    else:
        logging.warning("Perfecting stage reached iteration limit.")

    return [list(d) for d in current], current_score
