from __future__ import annotations
import logging
import networkx as nx
from shapely.strtree import STRtree




def build_adjacency_graph(gdf, progress_interval: int = 5000):
logging.info("Building adjacency graph with spatial index...")
G = nx.Graph()
for _, row in gdf.iterrows():
G.add_node(row['GEOID20'], pop=int(row['P1_001N']), x=row['x'], y=row['y'], geom=row.geometry)


geoms = list(gdf.geometry.values)
geoids = list(gdf['GEOID20'].values)
tree = STRtree(geoms)
geom_to_geoid = {geom: gid for geom, gid in zip(geoms, geoids)}


total = len(geoms)
for idx, geom in enumerate(geoms):
if (idx % progress_interval) == 0:
logging.info(f"Adjacency: processing geometry {idx}/{total}...")
for nbr in tree.query(geom):
if nbr is geom: continue
try:
if geom.touches(nbr):
G.add_edge(geom_to_geoid[geom], geom_to_geoid[nbr])
except Exception as ex:
logging.debug(f"Geometry touch test failed: {ex}")
continue
logging.info(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
return G
