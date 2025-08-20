from __future__ import annotations

import logging
import matplotlib.pyplot as plt


def plot_districts(gdf, final_districts, state_name: str, state_code: str):
    """
    Generate and save a color-coded district map PNG.
    """
    logging.info("Generating district map visualization...")
    gdf = gdf.copy()
    gdf["district"] = -1
    for i, d in enumerate(final_districts):
        gdf.loc[gdf["GEOID20"].isin(d), "district"] = i + 1

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(column="district", cmap="tab20", ax=ax, legend=True)
    plt.title(f"District Map for {state_name}")
    out = f"districts_{state_code}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    logging.info(f"District map saved to {out}")
