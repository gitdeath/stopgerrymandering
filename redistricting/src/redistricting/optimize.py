from __future__ import annotations

import logging
import random
from collections import Counter
import networkx as nx
from functools import partial
import multiprocessing

from .metrics import objective, is_contiguous
from .viz import plot_districts, print_debug_stats


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


def _evaluate_move_worker(block, current_districts, membership, G, gdf, ideal_pop, pop_tolerance_ratio):
    """
    Worker function for parallel processing in the Slow Pass.
    """
    best_move_for_block = None
    best_delta_for_block = 0.0
    
    from_idx = membership.get(block)
    if from_idx is None:
        return None, 0.0

    current_score = objective(current_districts, gdf, G, ideal_pop, pop_tolerance_ratio)
    neighbor_districts = {membership.get(n) for n in G.neighbors(block) if membership.get(n) is not None and membership.get(n) != from_idx}

    for to_idx in neighbor_districts:
        trial = [set(d) for d in current_districts]
        trial[from_idx].remove(block)
        trial[to_idx].add(block)

        if not is_contiguous(trial[from_idx], G):
            continue

        new_score = objective(trial, gdf, G, ideal_pop, pop_tolerance_ratio)
        delta = current_score - new_score

        if delta > best_delta_for_block:
            best_delta_for_block = delta
            best_move_for_block = (block, from_idx, to_idx)
            
    return best_move_for_block, best_delta_for_block


def perfect_map(districts, gdf, G, ideal_pop, pop_tolerance_ratio, st, scode, debug):
    """
    Final optimizer with a "good enough" Fast Pass and a parallelized Slow Pass.
    """
    current = [set(d) for d in districts]
    
    # --- PASS 1: EXHAUSTIVE, TARGETED FAST PASS ---
    logging.info("Perfecting Pass 1: Running exhaustive, population-only optimization...")
    while True:
        pop_per_district = [sum(G.nodes[b]["pop"] for b in d) for d in current]
        min_pop = ideal_pop * (1 - pop_tolerance_ratio)
        max_pop = ideal_pop * (1 + pop_tolerance_ratio)
        
        imbalanced_districts_with_dev = [
            (abs(p - ideal_pop), i) for i, p in enumerate(pop_per_district) 
            if not (min_pop <= p <= max_pop)
        ]

        if not imbalanced_districts_with_dev:
            logging.info("Fast Pass complete: All districts are within population tolerance.")
            break

        imbalanced_districts_with_dev.sort(key=lambda x: x[0], reverse=True)
        
        move_made_this_pass = False
        membership = {b: i for i, d in enumerate(current) for b in d}

        for _, district_to_fix_idx in imbalanced_districts_with_dev:
            is_over_populated = pop_per_district[district_to_fix_idx] > ideal_pop
            
            search_space = []
            if is_over_populated:
                search_space = list({b for b in current[district_to_fix_idx] if any(membership.get(n) != district_to_fix_idx for n in G.neighbors(b))})
            else:
                neighbors_of_worst = {n for b in current[district_to_fix_idx] for n in G.neighbors(b) if membership.get(n) != district_to_fix_idx}
                search_space = list(neighbors_of_worst)

            random.shuffle(search_space)

            for block in search_space:
                from_idx, to_idx = -1, -1
                if is_over_populated:
                    from_idx = district_to_fix_idx
                    candidates = {membership.get(n) for n in G.neighbors(block) if membership.get(n) is not None and n not in current[from_idx]}
                    if not candidates: continue
                    to_idx = min(candidates, key=lambda n_idx: pop_per_district[n_idx])
                else: 
                    to_idx = district_to_fix_idx
                    from_idx = membership.get(block)
                    if from_idx is None: continue

                # --- THE "GOOD ENOUGH" FIX ---
                # If both districts are already in the legal zone, don't make the move.
                from_pop = pop_per_district[from_idx]
                to_pop = pop_per_district[to_idx]
                if (min_pop <= from_pop <= max_pop) and (min_pop <= to_pop <= max_pop):
                    continue
                # --- END FIX ---
                
                block_pop = G.nodes[block]["pop"]
                old_score = (abs(from_pop - ideal_pop))**2 + (abs(to_pop - ideal_pop))**2
                new_score = (abs(from_pop - block_pop - ideal_pop))**2 + (abs(to_pop + block_pop - ideal_pop))**2
                
                if new_score < old_score:
                    if is_contiguous(current[from_idx] - {block}, G):
                        current[from_idx].remove(block)
                        current[to_idx].add(block)
                        move_made_this_pass = True
                        logging.debug(f"  - Fast Pass move applied: D{from_idx+1} -> D{to_idx+1}.")
                        break 
            
            if move_made_this_pass:
                break 
        
        if not move_made_this_pass:
            logging.info(f"Fast Pass: A full pass on all imbalanced districts found no improving moves. Concluding.")
            break
            
    if debug:
        plot_districts(gdf, current, st.name, scode, output_filename=f"debug_3b_fastpass_{scode}.png")
        print_debug_stats("Fast Pass Balancing", current, gdf, G)

    # --- PASS 2: PARALLELIZED SLOW SHAPE POLISHING ---
    logging.info("Perfecting Pass 2: Running parallelized, full hybrid-score optimization...")
    
    MAX_SLOW_PASS_ITERATIONS = 30
    for iteration in range(1, MAX_SLOW_PASS_ITERATIONS + 1):
        current_score = objective(current, gdf, G, ideal_pop, pop_tolerance_ratio)
        logging.info(f"Slow Pass Iteration {iteration}: Starting score {current_score:.2f}")

        membership = {b: i for i, d in enumerate(current) for b in d}
        all_border_blocks = list({b for d in current for b in d if any(membership.get(n) is not None and membership.get(n) != membership[b] for n in G.neighbors(b))})

        SAMPLE_SIZE = 500
        blocks_to_check = random.sample(all_border_blocks, min(len(all_border_blocks), SAMPLE_SIZE))
        
        logging.debug(f"  - Distributing analysis of {len(blocks_to_check)} blocks across CPU cores...")

        task = partial(
            _evaluate_move_worker, 
            current_districts=current,
            membership=membership, 
            G=G, 
            gdf=gdf, 
            ideal_pop=ideal_pop, 
            pop_tolerance_ratio=pop_tolerance_ratio,
        )

        best_move = None
        best_delta = 0.0
        
        with multiprocessing.Pool() as pool:
            results = pool.map(task, blocks_to_check)

        for move, delta in results:
            if delta > best_delta:
                best_delta = delta
                best_move = move
        
        if best_move:
            b, fidx, tidx = best_move
            current[fidx].remove(b)
            current[tidx].add(b)
            logging.info(f"  - Applied best move. New hybrid score: {current_score - best_delta:.2f}")
        else:
            logging.info("Slow Pass: No further shape improvements found.")
            break
            
    logging.info("Final perfecting complete.")
    return [list(d) for d in current], objective(current, gdf, G, ideal_pop, pop_tolerance_ratio)
