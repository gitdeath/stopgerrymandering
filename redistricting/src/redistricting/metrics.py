from __future__ import annotations
import numpy as np
import networkx as nx
from shapely.geometry import Polygon




def compute_inertia(district_blocks, gdf):
district_gdf = gdf[gdf['GEOID20'].isin(district_blocks)]
total_pop = district_gdf['P1_001N'].astype(int).sum()
if total_pop == 0: return float('inf')
cx = (district_gdf['P1_001N'].astype(int) * district_gdf['x']).sum() / total_pop
cy = (district_gdf['P1_001N'].astype(int) * district_gdf['y']).sum() / total_pop
return (
(district_gdf['P1_001N'].astype(int) * ((district_gdf['x'] - cx) ** 2 + (district_gdf['y'] - cy) ** 2)).sum()
)




def polsby_popper(district_blocks, gdf):
district_gdf = gdf[gdf['GEOID20'].isin(district_blocks)]
union_geom = district_gdf.geometry.unary_union
if isinstance(union_geom, Polygon):
area = union_geom.area
perim = union_geom.length
return (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0
return 0




def is_contiguous(district_blocks, G):
return nx.is_connected(G.subgraph(district_blocks))




def objective(districts, gdf, G, ideal_pop: float, pop_tolerance_ratio: float, compactness_threshold: float):
total_inertia = 0
pop_tol = ideal_pop * pop_tolerance_ratio
for d in districts:
if not d: return float('inf')
district_pop = sum(int(G.nodes[b]['pop']) for b in d)
if not (ideal_pop - pop_tol <= district_pop <= ideal_pop + pop_tol):
return float('inf')
if not is_contiguous(d, G):
return float('inf')
if polsby_popper(d, gdf) < compactness_threshold:
return float('inf')
total_inertia += compute_inertia(d, gdf)
return total_inertia
