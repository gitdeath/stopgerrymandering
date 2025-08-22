from __future__ import annotations

import logging
import numpy as np
import networkx as nx
from shapely.geometry import Polygon, MultiPolygon


def compute_inertia(district_blocks, gdf) -> float:
    """
    (This function is unchanged)
    """
    district_gdf = gdf[gdf["GEOID20"].isin(district_blocks)]
    total_pop = district_gdf["P1_001N"].astype(int).sum()
    if total_pop == 0:
        return 0.0
    cx = (district_gdf["P1_001N"].astype(int) * district_gdf["x"]).sum() / total_pop
    cy = (district_gdf["P1_001N"].astype(int) * district_gdf["y"]).sum() / total_pop
    inertia = (
        district_gdf["P1_001N"].astype(int)
        * ((district_gdf["x"] - cx) ** 2 + (district_gdf["y"] - cy) ** 2)
    ).sum()
    return float(inertia)


def polsby_popper(district_blocks, gdf) -> float:
    """
    (This is the corrected Polsby-Popper function)
    Polsby–Popper compactness: 4πA / P^2 for the unioned district geometry.
    This version correctly handles MultiPolygon objects.
    """
    district_gdf = gdf[gdf["GEOID20"].isin(district_blocks)]
    union_geom = district_gdf.geometry.unary_union
    
    if union_geom.is_empty:
        return 0.0

    # This calculation works for both Polygon and MultiPolygon objects
    area = union_geom.area
    perim = union_geom.length
    
    return float((4 * np.pi * area) / (perim ** 2)) if perim > 0 else 0.0


def is_contiguous(district_blocks, G: nx.Graph) -> bool:
    """(This function is unchanged)"""
    if not district_blocks:
        return False
    sub = G.subgraph(district_blocks)
    return nx.is_connected(sub)


def objective(
    districts,
    gdf,
    G: nx.Graph,
    ideal_pop: float,
    pop_tolerance_ratio: float,
    compactness_threshold: float,
) -> float:
    """
    (This function is unchanged)
    """
    total_inertia = 0.0
    population_penalty = 0.0
    contiguity_penalty = 0.0
    
    CONTIGUITY_PENALTY_WEIGHT = 1e18
    POPULATION_PENALTY_WEIGHT = 1e15

    min_pop = ideal_pop * (1 - pop_tolerance_ratio)
    max_pop = ideal_pop * (1 + pop_tolerance_ratio)

    for i, d in enumerate(districts):
        if not d:
            population_penalty += min_pop ** 2
            continue
        
        sub = G.subgraph(d)
        num_components = nx.number_connected_components(sub)
        if num_components > 1:
            contiguity_penalty += (num_components - 1)

        district_pop = sum(int(G.nodes[b]["pop"]) for b in d)
        if district_pop < min_pop:
            population_penalty += (min_pop - district_pop) ** 2
        elif district_pop > max_pop:
            population_penalty += (district_pop - max_pop) ** 2
        
        if num_components == 1:
            total_inertia += compute_inertia(d, gdf)
        
    final_score = (
        total_inertia +
        (population_penalty * POPULATION_PENALTY_WEIGHT) +
        (contiguity_penalty * CONTIGUITY_PENALTY_WEIGHT)
    )

    return float(final_score)
