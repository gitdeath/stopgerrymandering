from __future__ import annotations

import logging
import networkx as nx
from shapely.strtree import STRtree


def build_adjacency_graph(gdf, progress_interval: int = 5000):
    """
    Build a block-level adjacency graph using a spatial index.

    Nodes: GEOID20 (attrs: pop, x, y, geom)
    Edges: blocks that 'touch' (share boundary/vertex) per Shapely .touches()
    """
    logging.info("Building adjacency graph with spatial index...")

    G = nx.Graph()
    # Add nodes with attributes
    for _, row in gdf.iterrows():
        G.add_node(
            row["GEOID20"],
            pop=int(row["P1_001N"]),
            x=row["x"],
            y=row["y"],
            geom=row.geometry,
        )

    # Prepare geometry array (keep order aligned with GEOIDs)
    geoms = list(gdf.geometry.values)
    geoids = list(gdf["GEOID20"].values)

    # Filter out invalid/empty geometries by index map
    # (We keep arrays the same length; we’ll check validity on access.)
    tree = STRtree(geoms)

    # Use query_bulk for robust index pairs
    # pairs is shape (2, M) → first row: indices of query geoms, second row: indices of candidate geoms
    pairs = tree.query_bulk(geoms)  # all-to-all candidate pairs (spatial bbox hits)

    n_pairs = pairs.shape[1]
    logging.info(f"Spatial candidates: ~{n_pairs:,} bbox-hit pairs; testing touches...")

    # Build edges by testing 'touches' on candidate pairs
    # Only process each undirected pair once (i < j)
    for k in range(n_pairs):
        i = int(pairs[0, k])
        j = int(pairs[1, k])
        if i == j or i > j:
            continue  # skip self and duplicates

        gi = geoms[i]
        gj = geoms[j]

        # Skip empties/invalids defensively
        if gi is None or gj is None:
            continue
        if getattr(gi, "is_empty", False) or getattr(gj, "is_empty", False):
            continue

        try:
            # touches() is the exact topological rule we want
            if gi.touches(gj):
                G.add_edge(geoids[i], geoids[j])
        except Exception as ex:
            # Swallow rare type/GEOS issues without spamming logs
            # (Enable debug here only if you need to diagnose a specific dataset.)
            # logging.debug(f"touches() failed for pair ({i},{j}): {ex}")
            continue

        if progress_interval and (k % progress_interval) == 0:
            logging.info(f"Touch-test progress: {k}/{n_pairs} pairs...")

    logging.info(
        f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
    )
    return G
