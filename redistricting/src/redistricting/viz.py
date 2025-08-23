from __future__ import annotations

import logging
import matplotlib.pyplot as plt
# --- THE FIX (Step 1): Add the required import ---
import matplotlib.patheffects as path_effects


def plot_districts(
    gdf, 
    final_districts, 
    state_name: str, 
    state_code: str, 
    output_filename: str | None = None
):
    """
    Generate and save a color-coded and labeled district map PNG.
    """
    logging.info(f"Generating district map visualization...")
    gdf = gdf.copy()
    gdf["district"] = -1
    for i, d in enumerate(final_districts):
        district_blocks = list(d)
        gdf.loc[gdf["GEOID20"].isin(district_blocks), "district"] = i + 1

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(column="district", cmap="tab20", ax=ax, legend=True, categorical=True)
    
    # Calculate the center of each district and add a text label
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
                # --- THE FIX (Step 2): Correct the function call ---
                path_effects=[
                    path_effects.withStroke(linewidth=2, foreground='black')
                ]
            )

    if "debug" in (output_filename or ""):
        try:
            # Adjusted split for new debug filenames like "debug_1_initial_MO.png"
            phase_name = output_filename.split('_')[2].capitalize()
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
