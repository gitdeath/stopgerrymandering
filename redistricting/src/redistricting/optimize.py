from __future__ import annotations

import logging
import random
from collections import Counter
import networkx as nx

from .metrics import objective, is_contiguous, compute_inertia


def fix_contiguity(
    districts,
    gdf,
    G: nx.Graph,
):
    """
    Stage 1: Finds and repairs non-contiguous districts by reassigning island blocks.
    """
    fixed_districts = [set(d) for d in districts]
    
    while True:
        membership = {b: i for i, d in enumerate(fixed_districts) for b in d}
        fixes_made = 0
        for i, d in enumerate(fixed_districts):
            if not d: continue
            
            subgraph = G.subgraph(d)
            components = list(nx.connected_components(subgraph))
            
            if len(components) > 1:
                components.sort(key=len, reverse=True)
                islands = components[1:]
                
                for island in islands:
                    for block in island:
                        neighbors_in_main_graph = G.neighbors(block)
                        
                        neighbor_districts = [
                            membership[n] for n in neighbors_in_main_graph 
                            if n in membership and membership[n] != i
                        ]
                        
                        if neighbor_districts:
                            most_common_neighbor_dist = Counter(neighbor_districts).most_common(1)[0][0]
                            
                            logging.warning(
                                f"CONTIGUITY FIX: Moving block {block} from District {i+1} "
                                f"to its neighbor, District {most_common_neighbor_dist+1}."
                            )
                            
                            fixed_districts[i].remove(block)
                            fixed_districts[most_common_neighbor_dist].add(block)
                            fixes_made += 1

        if fixes_made == 0:
            break
            
    for i, d in enumerate(fixed_districts):
        if not is_contiguous(d, G):
            logging.error(f"FATAL: Could not fix contiguity for District {i+1}. Exiting.")
            raise RuntimeError(f"Contiguity fix failed for District {i+1}")

    logging.info("Contiguity repair complete.")
    return fixed_districts


def _calculate_district_score(d, gdf, G, ideal_pop, min_pop, max_pop):
    """Helper to calculate the objective score for a single district."""
    if not d:
        return (1e18 * (min_pop**2)) # Penalize empty districts

    # Hard contiguity constraint for individual scoring
    if not is_contiguous(d, G):
        return float('inf')

    inertia = compute_inertia(d, gdf)
    
    district_pop = sum(int(G.nodes[b]["pop"]) for b in d)
    pop_penalty = 0
    if district_pop < min_pop:
        pop_penalty = (min_pop - district_pop) ** 2
    elif district_pop > max_pop:
        pop_penalty = (district_pop - max_pop) ** 2
        
    return inertia + (pop_penalty * 1e15)


def rapid_balance(
    districts,
    gdf,
    G,
    ideal_pop: float,
    pop_tolerance_ratio: float,
    compactness_threshold: float,
    passes: int = 2,
):
    """
    Stage 2: A faster, "delta"-based optimization that applies good moves immediately.
    """
    current = [set(d) for d in districts]
    min_pop = ideal_pop * (1 - pop_tolerance_ratio)
    max_pop = ideal_pop * (1 + pop_tolerance_ratio)

    # Calculate the initial score fully once
    current_score = objective(current, gdf, G, ideal_pop, pop_tolerance_ratio, compactness_threshold)
    logging.info(f"Rapid balancing starting score: {current_score:.2f}")

    all_blocks = list(G.nodes)
    
    for p in range(passes):
        random.Random(p).shuffle(all_blocks)
        moves_made_in_pass = 0
        
        membership = {b: i for i, d in enumerate(current) for b in d}

        for i, block in enumerate(all_blocks):
            from_idx = membership.get(block)
            if from_idx is None: continue

            neighbor_districts = {
                membership.get(n) for n in G.neighbors(block)
                if membership.get(n) is not None and membership[n] != from_idx
            }

            for to_idx in neighbor_districts:
                # --- DELTA SCORING ---
                # 1. Get the score of the two affected districts BEFORE the move
                score_before = _calculate_district_score(current[from_idx], gdf, G, ideal_pop, min_pop, max_pop)
                score_before += _calculate_district_score(current[to_idx], gdf, G, ideal_pop, min_pop, max_pop)

                # 2. Create hypothetical new districts
                new_from_district = current[from_idx] - {block}
                new_to_district = current[to_idx] | {block}
                
                # 3. Get the score of the two affected districts AFTER the move
                score_after = _calculate_district_score(new_from_district, gdf, G, ideal_pop, min_pop, max_pop)
                score_after += _calculate_district_score(new_to_district, gdf, G, ideal_pop, min_pop, max_pop)
                
                # 4. If the local change is an improvement, apply it.
                if score_after < score_before:
                    current[from_idx] = new_from_district
                    current[to_idx] = new_to_district
                    membership[block] = to_idx # Update membership for the moved block
                    moves_made_in_pass += 1
                    break 

            if (i + 1) % 5000 == 0:
                logging.info(f"Rapid balance pass {p+1}/{passes}: scanned {i+1}/{len(all_blocks)} blocks, found {moves_made_in_pass} moves.")
        
        # Recalculate full score at the end of the pass for an accurate progress report
        current_score = objective(current, gdf, G, ideal_pop, pop_tolerance_ratio, compactness_threshold)
        logging.info(f"Rapid balance pass {p+1} complete. New score: {current_score:.2f}. Total moves made: {moves_made_in_pass}")
        if moves_made_in_pass == 0:
            break

    return current, current_score


def perfect_map(
    districts,
    gdf,
    G,
    ideal_pop: float,
    pop_tolerance_ratio: float,
    compactness_threshold: float,
):
    """
    Stage 3: The original, slow, and meticulous optimization to find the local optimum.
    """
    current = [set(d) for d in districts]
    current_score = objective(current, gdf, G, ideal_pop, pop_tolerance_ratio, compactness_threshold)
    logging.info(f"Final perfecting starting score: {current_score:.2f}")

    iteration = 0
    while True:
        iteration += 1
        best_move = None
        best_delta = 0.0

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
                    best_delta = delta
                    best_move = (block, from_idx, to_idx)

        if best_move:
            b, fidx, tidx = best_move
            current[fidx].remove(b)
            current[tidx].add(b)
            current_score -= best_delta
            logging.info(
                f"Perfecting Iteration {iteration}: Applied best move; new score {current_score:.2f}"
            )
        else:
            logging.info(
                f"Perfecting Iteration {iteration}: No further improvements found. Optimization complete."
            )
            break

    return current, current_score
