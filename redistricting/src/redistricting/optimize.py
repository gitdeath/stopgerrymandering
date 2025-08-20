from __future__ import annotations
import logging
from .metrics import objective




def optimize_districts(districts, gdf, G, ideal_pop: float, pop_tolerance_ratio: float, compactness_threshold: float):
logging.info("Step 4 of 5: Optimizing district map using a greedy approach...")
current = [set(d) for d in districts]
current_score = objective(current, gdf, G, ideal_pop, pop_tolerance_ratio, compactness_threshold)
logging.info(f"Initial objective score: {current_score:.2f}")


iteration = 0
while True:
iteration += 1
best_move = None
best_delta = 0
all_blocks = list(G.nodes)


# Try moving each block to a neighboring district only
membership = {b: i for i, d in enumerate(current) for b in d}
for block in all_blocks:
from_idx = membership.get(block)
if from_idx is None: continue
neighbor_districts = {membership[n] for n in G.neighbors(block) if membership.get(n) is not None and membership[n] != from_idx}
for to_idx in neighbor_districts:
trial = [set(d) for d in current]
trial[from_idx].remove(block)
trial[to_idx].add(block)
new_score = objective(trial, gdf, G, ideal_pop, pop_tolerance_ratio, compactness_threshold)
if new_score < current_score and (current_score - new_score) > best_delta:
best_delta = current_score - new_score
best_move = (block, from_idx, to_idx)
if best_move:
b, fidx, tidx = best_move
current[fidx].remove(b)
current[tidx].add(b)
current_score -= best_delta
logging.info(f"Iteration {iteration}: Applied best move. New score: {current_score:.2f}")
else:
logging.info(f"Iteration {iteration}: No further improvements found. Optimization complete.")
break
return current, current_score
