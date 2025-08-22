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
    Generate and save a color-coded district map PNG.
    If output_filename is provided, it's used; otherwise, a default is created.
    """
    logging.info(f"Generating district map visualization...")
    gdf = gdf.copy()
    gdf["district"] = -1
    for i, d in enumerate(final_districts):
        # Ensure d is a list or set of GEOID20s before using .isin
        district_blocks = list(d)
        gdf.loc[gdf["GEOID20"].isin(district_blocks), "district"] = i + 1

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(column="district", cmap="tab20", ax=ax, legend=True)
    
    # Determine the title and filename
    if "debug" in (output_filename or ""):
        try:
            phase_name = output_filename.split('_')[1].capitalize()
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
