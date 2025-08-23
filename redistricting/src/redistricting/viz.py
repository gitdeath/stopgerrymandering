from __future__ import annotations

import logging
import matplotlib.pyplot as plt


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
    
    # --- NEW LABELING LOGIC ---
    # Calculate the center of each district and add a text label
    for i in range(1, len(final_districts) + 1):
        district_gdf = gdf[gdf["district"] == i]
        if not district_gdf.empty:
            # Calculate the centroid of the union of all blocks in the district
            centroid = district_gdf.geometry.unary_union.centroid
            # Add the district number as a text label
            ax.text(
                centroid.x, 
                centroid.y, 
                str(i), 
                fontsize=12, 
                fontweight='bold',
                ha='center', 
                va='center',
                color='white',
                # Add a black outline to the text for better visibility
                path_effects=[
                    plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='black')
                ]
            )
    # --- END NEW LABELING LOGIC ---

    if "debug" in (output_filename or ""):
        try:
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
