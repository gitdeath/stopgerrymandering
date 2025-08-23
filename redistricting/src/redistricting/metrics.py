from __future__ import annotations

import logging
import numpy as np
import networkx as nx
from shapely.geometry import Polygon, MultiPolygon


def compute_inertia(district_blocks, gdf) -> float:
    """
    Population-weighted moment of inertia about the district's pop-weighted centroid.
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

def compute_bounding_box_score(district_blocks, gdf) -> float:
    """
    Measures how well a district fills its rectangular bounding box.
    Score of 1.0 is a perfect rectangle. Lower is worse.
    """
    district_gdf = gdf[gdf["GEOID20"].isin(district_blocks)]
    union_geom = district_gdf.geometry.unary_union
    
    if union_geom.is_empty:
        return 0.0

    district_area = union_geom.area
    bounding_box = union_geom.envelope # The minimum bounding rectangle
    box_area = bounding_box.area

    if box_area == 0:
        return 0.0
        
    return district_area / box_area


def polsby_popper(district_blocks, gdf) -> float:
    """
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
    """A district is contiguous if its induced subgraph is connected."""
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
) -> float:
    """
    The hybrid objective function. It minimizes a weighted combination of
    inertia (population compactness) and a bounding box penalty (geometric compactness).
    """
    total_inertia_score = 0.0
    total_bbox_score = 0.0
    population_penalty = 0.0
    contiguity_penalty = 0.0
    
    # --- Huge penalties for illegal maps (Hard Constraints) ---
    CONTIGUITY_PENALTY_WEIGHT = 1e18
    POPULATION_PENALTY_WEIGHT = 1e15

    # --- Weights for the hybrid score (Soft Constraints) ---
    INERTIA_WEIGHT = 1.0
    BBOX_WEIGHT = 10000.0 

    min_pop = ideal_pop * (1 - pop_tolerance_ratio)
    max_pop = ideal_pop * (1 + pop_tolerance_ratio)

    for i, d in enumerate(districts):
        if not d:
            population_penalty += min_pop ** 2
            continue
        
        # --- 1. Contiguity Penalty (Highest Priority) ---
        sub = G.subgraph(d)
        num_components = nx.number_connected_components(sub)
        if num_components > 1:
            contiguity_penalty += (num_components - 1)

        # --- 2. Population Penalty (Second Priority) ---
        district_pop = sum(int(G.nodes[b]["pop"]) for b in d)
        if district_pop < min_pop:
            population_penalty += (min_pop - district_pop) ** 2
        elif district_pop > max_pop:
            population_penalty += (district_pop - max_pop) ** 2
        
        # --- 3. Hybrid Compactness Score (Base Score) ---
        if num_components == 1:
            inertia = compute_inertia(d, gdf)
            total_inertia_score += np.log(1 + inertia) if inertia > 0 else 0

            bbox_score = compute_bounding_box_score(d, gdf)
            total_bbox_score += (1 - bbox_score)
        
    final_score = (
        (total_inertia_score * INERTIA_WEIGHT) +
        (total_bbox_score * BBOX_WEIGHT) +
        (population_penalty * POPULATION_PENALTY_WEIGHT) +
        (contiguity_penalty * CONTIGUITY_PENALTY_WEIGHT)
    )

    return float(final_score)
