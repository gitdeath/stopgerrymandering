from __future__ import annotations

import logging
import random
from collections import Counter
import networkx as nx

from .metrics import objective, is_contiguous


def fix_contiguity(districts, gdf, G: nx.Graph):
    # This function is correct and does not need changes.
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
    # This function is correct and does not need changes.
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


def _calculate_fast_score(districts, G, ideal_pop):
    """A simplified, fast score that only considers population deviation."""
    score = 0
    for d in districts:
        if not d: continue
        pop = sum(G.nodes[b]["pop"] for b in d)
        score += (pop - ideal_pop) ** 2
    return score


def perfect_map(districts, gdf, G, ideal_pop, pop_tolerance_ratio):
    """
    Final optimizer with a two-pass 'fast first' approach. The fast pass uses
    a targeted search on the most imbalanced district. The slow pass polishes
    the full map with the hybrid score.
    """
    logging.info("Starting Stage 3: Final Perfecting...")
    current = [set(d) for d in districts]
    
    # --- PASS 1: TARGETED FAST PASS ---
    logging.info("Perfecting Pass 1: Running targeted, population-only optimization...")
    MAX_FAST_PASS_ITERATIONS = 100 
    for iteration in range(1, MAX_FAST_PASS_ITERATIONS + 1):
        
        pop_per_district = [sum(G.nodes[b]["pop"] for b in d) for d in current]
        pop_devs = [(abs(p - ideal_pop), i) for i, p in enumerate(pop_per_district)]
        
        # --- NEW: Identify the district that is most "out of bounds" ---
        worst_dev, worst_idx = max(pop_devs)
        logging.debug(f"Fast Pass Iteration {iteration}: Targeting D{worst_idx+1} (deviation: {worst_dev:,.0f}).")
        
        current_score = (worst_dev) ** 2 # We only need to track the score of the district we're fixing
        best_move, move_applied = None, False
        membership = {b: i for i, d in enumerate(current) for b in d}
        
        # --- NEW: Only check blocks on the border of the worst district ---
        border_blocks_of_worst_dist = list({
            b for b in current[worst_idx] 
            if any(membership.get(n) is not None and membership.get(n) != worst_idx for n in G.neighbors(b))
        })
        
        SAMPLE_SIZE = 750
        blocks_to_check = random.sample(border_blocks_of_worst_dist, min(len(border_blocks_of_worst_dist), SAMPLE_SIZE))

        for block in blocks_to_check:
            from_idx = worst_idx
            for to_idx in {membership.get(n) for n in G.neighbors(block) if membership.get(n) is not None and membership.get(n) != from_idx}:
                trial = [set(d) for d in current]
                trial[from_idx].remove(block)
                trial[to_idx].add(block)
                if not is_contiguous(trial[from_idx], G): continue
                
                new_score = (abs(sum(G.nodes[b]["pop"] for b in trial[from_idx]) - ideal_pop))**2 + \
                            (abs(sum(G.nodes[b]["pop"] for b in trial[to_idx]) - ideal_pop))**2
                old_score = (abs(pop_per_district[from_idx] - ideal_pop))**2 + \
                            (abs(pop_per_district[to_idx] - ideal_pop))**2

                if new_score < old_score:
                    # --- NEW: "First Improvement" Strategy ---
                    best_move = (block, from_idx, to_idx)
                    current[from_idx].remove(best_move[0])
                    current[to_idx].add(best_move[0])
                    move_applied = True
                    logging.debug(f"  - Fast Pass move applied.")
                    break # Apply first good move found
            if move_applied:
                break # Start new iteration

        if not move_applied:
            logging.info("Fast Pass: No further population improvements found for the worst district.")
            break

    # --- PASS 2: SLOW SHAPE POLISHING ---
    logging.info("Perfecting Pass 2: Running full hybrid-score optimization...")
    current_score = objective(current, gdf, G, ideal_pop, pop_tolerance_ratio)
    logging.info(f"Hybrid score after fast pass: {current_score:.2f}")

    MAX_SLOW_PASS_ITERATIONS = 30
    for iteration in range(1, MAX_SLOW_PASS_ITERATIONS + 1):
        best_move, best_delta = None, 0.0
        membership = {b: i for i, d in enumerate(current) for b in d}
        all_border_blocks = list({b for d in current for b in d if any(membership.get(n) is not None and membership.get(n) != membership[b] for n in G.neighbors(b))})

        SAMPLE_SIZE = 500
        blocks_to_check = random.sample(all_border_blocks, min(len(all_border_blocks), SAMPLE_SIZE))
        
        logging.debug(f"Slow Pass Iteration {iteration}: Checking {len(blocks_to_check)} blocks for shape improvements...")

        for block in blocks_to_check:
            from_idx = membership.get(block)
            if from_idx is None: continue
            for to_idx in {membership.get(n) for n in G.neighbors(block) if membership.get(n) is not None and membership.get(n) != from_idx}:
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
            logging.info(f"Slow Pass Iteration {iteration}: Applied best move. New hybrid score: {current_score:.2f}")
        else:
            logging.info("Slow Pass: No further shape improvements found.")
            break
            
    logging.info("Final perfecting complete.")
    return [list(d) for d in current], current_score
