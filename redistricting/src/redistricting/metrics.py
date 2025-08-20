from __future__ import annotations

import numpy as np
import networkx as nx
from shapely.geometry import Polygon


def compute_inertia(district_blocks, gdf) -> float:
    """
    Population-weighted moment of inertia about the district's pop-weighted centroid.
    Returns +inf if district_pop == 0 (invalid).
    """
    district_gdf = gdf[gdf["GEOID20"].isin(district_blocks)]
    total_pop = district_gdf["P1_001N"].astype(int).sum()
    if total_pop == 0:
        return float("inf")

    cx = (district_gdf["P1_001N"].astype(int) * district_gdf["x"]).sum() / total_pop
    cy = (district_gdf["P1_001N"].astype(int) * district_gdf["y"]).sum() / total_pop

    inertia = (
        district_gdf["P1_001N"].astype(int)
        * ((district_gdf["x"] - cx) ** 2 + (district_gdf["y"] - cy) ** 2)
    ).sum()

    return float(inertia)


def polsby_popper(district_blocks, gdf) -> float:
    """
    Polsby–Popper compactness: 4πA / P^2 for the unioned district geometry.
    Returns 0 if geometry is invalid or perimeter is 0.
    """
    district_gdf = gdf[gdf["GEOID20"].isin(district_blocks)]
    union_geom = district_gdf.geometry.unary_union
    if isinstance(union_geom, Polygon):
        area = union_geom.area
        perim = union_geom.length
        return float((4 * np.pi * area) / (perim ** 2)) if perim > 0 else 0.0
    # For multi-polygons or non-polygonal results, you could consider union.buffer(0)
    # and re-check, but we keep it simple/strict here.
    return 0.0


def is_contiguous(district_blocks, G: nx.Graph) -> bool:
    """A district is contiguous if its induced subgraph is connected."""
    sub = G.subgraph(district_blocks)
    # Empty or single-node districts are trivially connected only if size >= 1
    return nx.is_connected(sub) if sub.number_of_nodes() > 0 else False


def objective(
    districts,
    gdf,
    G: nx.Graph,
    ideal_pop: float,
    pop_tolerance_ratio: float,
    compactness_threshold: float,
) -> float:
    """
    Objective to minimize: sum of moments of inertia across districts,
    subject to:
      - population within ±pop_tolerance_ratio of ideal_pop
      - contiguity
      - Polsby–Popper >= compactness_threshold
    Returns +inf if any constraint is violated.
    """
    total_inertia = 0.0
    pop_tol = ideal_pop * pop_tolerance_ratio

    for d in districts:
        if not d:
            return float("inf")

        district_pop = sum(int(G.nodes[b]["pop"]) for b in d)
        if not (ideal_pop - pop_tol <= district_pop <= ideal_pop + pop_tol):
            return float("inf")

        if not is_contiguous(d, G):
            return float("inf")

        if polsby_popper(d, gdf) < compactness_threshold:
            return float("inf")

        total_inertia += compute_inertia(d, gdf)

    return float(total_inertia)
