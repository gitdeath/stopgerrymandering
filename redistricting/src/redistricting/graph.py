from __future__ import annotations

import logging
import networkx as nx


def build_adjacency_graph(gdf, progress_interval: int = 5000):
    """
    Build a block-level adjacency graph.

    Nodes: GEOID20 (attrs: pop, x, y, geom)
    Edges: blocks that 'touch' or 'intersect' (share boundary/vertex).
    Compatible with older/newer GeoPandas/Shapely/rtree backends.
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

    geoms = list(gdf.geometry.values)
    geoids = list(gdf["GEOID20"].values)

    sindex = gdf.sindex
    if sindex is None:
        raise RuntimeError(
            "GeoPandas spatial index unavailable. Install 'rtree' (or pygeos/shapely>=2)."
        )

    # --- Strategy selection -------------------------------------------------
    # Try the fastest route first (bulk with predicate='intersects'); else try bulk bbox;
    # else per-geometry query with predicate; else per-geometry bbox + postcheck.
    has_query_bulk = hasattr(sindex, "query_bulk")
    supports_predicate_bulk = False
    if has_query_bulk:
        try:
            # Probe predicate support without doing full work
            _ = sindex.query_bulk(gdf.geometry[:1], predicate="intersects")
            supports_predicate_bulk = True
        except Exception:
            supports_predicate_bulk = False

    # --- Add edges ----------------------------------------------------------
    if has_query_bulk and supports_predicate_bulk:
        # Fast path: bulk + predicate in the index
        pairs = sindex.query_bulk(gdf.geometry, predicate="intersects")
        n_pairs = pairs.shape[1]
        logging.info(f"Spatial candidates (intersects via index): {n_pairs:,} pairs")

        for k in range(n_pairs):
            i = int(pairs[0, k]); j = int(pairs[1, k])
            if i >= j: # Use >= to avoid self-loops and duplicates
                continue
            G.add_edge(geoids[i], geoids[j])
            if progress_interval and (k % progress_interval) == 0:
                logging.info(f"Edge-build progress: {k}/{n_pairs} pairs...")

    elif has_query_bulk:
        # Bulk bbox hits, then post-check intersects()
        pairs = sindex.query_bulk(gdf.geometry)
        n_pairs = pairs.shape[1]
        logging.info(f"Spatial candidates (bbox): {n_pairs:,} pairs; testing intersects...")

        for k in range(n_pairs):
            i = int(pairs[0, k]); j = int(pairs[1, k])
            if i >= j: # Use >= to avoid self-loops and duplicates
                continue
            gi = geoms[i]; gj = geoms[j]
            if not gi or not gj or getattr(gi, "is_empty", False) or getattr(gj, "is_empty", False):
                continue
            try:
                # *** KEY CHANGE: Use intersects() instead of touches() ***
                if gi.intersects(gj):
                    G.add_edge(geoids[i], geoids[j])
            except Exception:
                continue
            if progress_interval and (k % progress_interval) == 0:
                logging.info(f"Intersect-test progress: {k}/{n_pairs} pairs...")

    else:
        # Oldest fallback: per-geometry query
        n = len(geoms)
        # Check if per-geometry predicate is supported
        supports_predicate = False
        try:
            _ = sindex.query(gdf.geometry.iloc[0:1], predicate="intersects")
            supports_predicate = True
        except Exception:
            supports_predicate = False

        edges_added = 0
        for i, gi in enumerate(geoms):
            if i % max(1, progress_interval // 10) == 0:
                logging.info(f"Adjacency: processing geometry {i}/{n}...")

            if not gi or getattr(gi, "is_empty", False):
                continue

            try:
                if supports_predicate:
                    candidates = sindex.query(gdf.geometry.iloc[i:i+1], predicate="intersects")
                else:
                    candidates = sindex.query(gdf.geometry.iloc[i:i+1])  # bbox
                # candidates is an index-like; iterate over integer positions
                for j in getattr(candidates, "tolist", lambda: list(candidates))():
                    if i >= j: # Use >= to avoid self-loops and duplicates
                        continue
                    gj = geoms[j]
                    if not gj or getattr(gj, "is_empty", False):
                        continue
                    if supports_predicate:
                        # already 'intersects' by predicate
                        G.add_edge(geoids[i], geoids[j])
                        edges_added += 1
                    else:
                        # bbox-hit: post-check intersects
                        try:
                             # *** KEY CHANGE: Use intersects() instead of touches() ***
                            if gi.intersects(gj):
                                G.add_edge(geoids[i], geoids[j])
                                edges_added += 1
                        except Exception:
                            continue
            except Exception:
                # Defensive: skip odd cases to keep building
                continue

        logging.info(f"Edges added (per-geometry mode): {edges_added:,}")

    logging.info(
        f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
    )
    return G
