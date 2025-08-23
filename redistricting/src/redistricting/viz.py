from __future__ import annotations

import logging
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
# --- THE FIX: Add imports needed for the new function ---
import networkx as nx
from .metrics import polsby_popper, is_contiguous, compute_inertia


def plot_districts(
    gdf, 
    final_districts, 
    state_name: str, 
    state_code: str, 
    output_filename: str | None = None
):
    # This function is unchanged
    logging.info(f"Generating district map visualization...")
    gdf = gdf.copy()
    gdf["district"] = -1
    for i, d in enumerate(final_districts):
        district_blocks = list(d)
        gdf.loc[gdf["GEOID20"].isin(district_blocks), "district"] = i + 1

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(column="district", cmap="tab20", ax=ax, legend=True, categorical=True)
    
    for i in range(1, len(final_districts) + 1):
        district_gdf = gdf[gdf["district"] == i]
        if not district_gdf.empty:
            centroid = district_gdf.geometry.unary_union.centroid
            ax.text(
                centroid.x, 
                centroid.y, 
                str(i), 
                fontsize=12, 
                fontweight='bold',
                ha='center', 
                va='center',
                color='white',
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')]
            )

    if "debug" in (output_filename or ""):
        try:
            phase_name_parts = output_filename.split('_')[2:-1]
            phase_name = " ".join(part.capitalize() for part in phase_name_parts)
            plt.title(f"District Map for {state_name} (After {phase_name})")
        except IndexError:
            plt.title(f"Debug District Map for {state_name}")
    else:
        plt.title(f"Final District Map for {state_name}")

    if output_filename:
        out_path = output_filename
    else:
        out_path = f"districts_{state_code}.png"
        
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    logging.info(f"District map saved to {out_path}")

# --- THE FIX: The print_debug_stats function is PASTED into this file ---
def print_debug_stats(phase_name: str, districts, gdf, G):
    """Helper function to print district stats during debug runs."""
    header = f" DEBUG STATS: After {phase_name} "
    logging.info("\n" + header.center(80, "-"))
    
    # Ensure districts is a list of lists/sets for consistent processing
    districts_list = [list(d) for d in districts]
    
    for i, d in enumerate(districts_list):
        if not d:
            logging.info(f"District {i+1}: EMPTY")
            continue
            
        d_pop = sum(G.nodes[b]["pop"] for b in d)
        d_pp = polsby_popper(d, gdf)
        d_inertia = compute_inertia(d, gdf)
        d_contig = is_contiguous(d, G)
        logging.info(
            f"District {i+1}: Pop = {d_pop:<9,} | "
            f"PP = {d_pp:.4f} | "
            f"Inertia = {d_inertia:.2e} | "
            f"Contiguous = {d_contig}"
        )
    logging.info("-" * 80 + "\n")
