from __future__ import annotations
import hashlib
import numpy as np
import logging




def get_sweep_order(gdf):
block_ids = sorted(gdf['GEOID20'])
hash_val = hashlib.sha256(''.join(block_ids).encode()).hexdigest()
sweep_idx = int(hash_val, 16) % 4
if sweep_idx == 0:
return gdf.sort_values(by=['y', 'x'], ascending=[False, True])['GEOID20'].tolist()
elif sweep_idx == 1:
return gdf.sort_values(by=['y', 'x'], ascending=[True, True])['GEOID20'].tolist()
elif sweep_idx == 2:
return gdf.sort_values(by=['y', 'x'], ascending=[True, False])['GEOID20'].tolist()
else:
return gdf.sort_values(by=['y', 'x'], ascending=[False, False])['GEOID20'].tolist()




def initial_assignment(gdf, G, D: int, ideal_pop: float, pop_tolerance_ratio: float):
logging.info("Step 3 of 5: Generating initial district map...")
sweep_order = get_sweep_order(gdf)
districts = [set() for _ in range(D)]
pop_per_district = [0] * D
pop_tolerance = ideal_pop * pop_tolerance_ratio


for i, block in enumerate(sweep_order):
block_pop = int(G.nodes[block]['pop'])
best_candidates = []
for j in range(D):
is_adjacent = not districts[j] or any(G.has_edge(block, b) for b in districts[j])
if (pop_per_district[j] + block_pop <= ideal_pop + pop_tolerance and is_adjacent):
best_candidates.append((pop_per_district[j], j))
if best_candidates:
best_candidates.sort()
best_idx = best_candidates[0][1]
else:
best_idx = int(np.argmin(pop_per_district))
districts[best_idx].add(block)
pop_per_district[best_idx] += block_pop
if (i + 1) % 1000 == 0 or i == len(sweep_order) - 1:
logging.info(f"Assigned {i + 1} of {len(sweep_order)} blocks.")
return districts
