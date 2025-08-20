from __future__ import annotations

import logging
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
    return 0.0


def is_contiguous(district_blocks, G: nx.Graph) -> bool:
    """A district is contiguous if its induced subgraph is connected."""
    sub = G.subgraph(district_blocks)
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
    subject to constraints. Returns +inf if any constraint is violated.
    """
    total_inertia = 0.0
    pop_tol = ideal_pop * pop_tolerance_ratio

    for i, d in enumerate(districts):
        dist_num = i + 1
        if not d:
            logging.error(f"Constraint Failed: District {dist_num} is empty.")
            return float("inf")

        district_pop = sum(int(G.nodes[b]["pop"]) for b in d)
        min_pop, max_pop = ideal_pop - pop_tol, ideal_pop + pop_tol
        if not (min_pop <= district_pop <= max_pop):
            logging.error(
                f"Constraint Failed: District {dist_num} population is {district_pop:,}, "
                f"but must be between {min_pop:,.0f} and {max_pop:,.0f}."
            )
            return float("inf")

        if not is_contiguous(d, G):
            logging.error(f"Constraint Failed: District {dist_num} is not contiguous.")
            return float("inf")

        pp_score = polsby_popper(d, gdf)
        if pp_score < compactness_threshold:
            logging.error(
                f"Constraint Failed: District {dist_num} Polsby-Popper score is {pp_score:.3f}, "
                f"below threshold of {compactness_threshold:.3f}."
            )
            return float("inf")

        total_inertia += compute_inertia(d, gdf)

    return float(total_inertia)
