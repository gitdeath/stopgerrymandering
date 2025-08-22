from __future__ import annotations

import logging
import networkx as nx


def build_adjacency_graph(gdf, progress_interval: int = 5000):
    """
    Build a block-level adjacency graph using a simple, robust method.

    Nodes: GEOID20 (attrs: pop, x, y, geom)
    Edges: blocks that 'intersect' (share boundary/vertex).
    """
    logging.info("Building adjacency graph with spatial index...")

    G = nx.Graph()
    # Add nodes with attributes from the GeoDataFrame
    for _, row in gdf.iterrows():
        G.add_node(
            row["GEOID20"],
            pop=int(row["P1_001N"]),
            x=row["x"],
            y=row["y"],
            geom=row.geometry,
        )

    # Ensure the spatial index is available
    sindex = gdf.sindex
    if sindex is None:
        raise RuntimeError(
            "GeoPandas spatial index unavailable. Install 'rtree'."
        )

    logging.info("Finding adjacent blocks...")
    # Use a single, reliable method to find and add edges
    for i, row in gdf.iterrows():
        current_geoid = row["GEOID20"]
        current_geom = row.geometry

        # Use the spatial index to find potential neighbors (bounding box intersection)
        possible_matches_index = list(sindex.intersection(current_geom.bounds))

        # Filter for actual intersections and add edges
        for possible_match_index in possible_matches_index:
            # Don't check a block against itself or create duplicate edges
            if i >= possible_match_index:
                continue

            neighbor_row = gdf.iloc[possible_match_index]
            neighbor_geoid = neighbor_row["GEOID20"]
            neighbor_geom = neighbor_row.geometry

            if current_geom.intersects(neighbor_geom):
                G.add_edge(current_geoid, neighbor_geoid)

        if progress_interval and (i % progress_interval) == 0 and i > 0:
            logging.info(f"Adjacency progress: processed {i}/{len(gdf)} blocks...")
    
    logging.info(
        f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
    )
    return G
