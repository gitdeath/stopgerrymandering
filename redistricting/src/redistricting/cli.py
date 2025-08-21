from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Settings
from .io import unzip_and_find_files, load_and_preprocess_data
from .assign import initial_assignment
from .optimize import optimize_districts
from .metrics import polsby_popper
from .viz import plot_districts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Redistricting algorithm using local zipped files"
    )
    parser.add_argument(
        "--state", required=True, help="Two-letter state code (e.g., MO, CA)"
    )
    parser.add_argument(
        "--districts",
        type=int,
        help="Number of districts (default: congressional per config)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "config" / "states.yaml"),
        help="Path to YAML/JSON config (default: config/states.yaml in repo root)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debugging output.",
    )
    return parser.parse_args()


def setup_logging(debug: bool):
    log_filename = "redistricting_run.log"
    
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Ensures this configuration is applied
    )

    logging.info(f"Detailed log being saved to: {log_filename}")
    
def main():
    args = parse_args()
    setup_logging(args.debug)

    settings = Settings.load(args.config)
    scode = args.state.upper()
    if scode not in settings.states:
        raise ValueError(f"Invalid state code: {scode}. Use a two-letter code.")
    st = settings.states[scode]
    D = args.districts if args.districts else st.districts

    # repo root (two levels up from this file: src/redistricting/cli.py → repo root)
    base_dir = Path(__file__).resolve().parents[2]
    paths = unzip_and_find_files(base_dir, st.fips, scode)

    gdf, G, total_pop = load_and_preprocess_data(paths, settings.defaults.crs_epsg)
    ideal_pop = total_pop / D
    logging.info(
        f"Starting redistricting for {st.name} with {D} districts. "
        f"Ideal pop: {ideal_pop:.2f}"
    )

    initial = initial_assignment(
        gdf,
        G,
        D,
        ideal_pop,
        pop_tolerance_ratio=settings.defaults.pop_tolerance_ratio,
    )

    final, score = optimize_districts(
        initial,
        gdf,
        G,
        ideal_pop,
        pop_tolerance_ratio=settings.defaults.pop_tolerance_ratio,
        compactness_threshold=settings.defaults.compactness_threshold,
    )

    # Tie-break / stable order
    final_sorted = [sorted(list(d)) for d in final]
    final_sorted.sort()

    # Visualization
    plot_districts(gdf, final_sorted, st.name, scode)

    # Console summary
    final_pop_counts = [sum(G.nodes[b]["pop"] for b in d) for d in final_sorted]
    print("\n" + "=" * 50)
    print(f"Final District Map for {st.name} ({D} districts)")
    print("=" * 50)
    for i, d_pop in enumerate(final_pop_counts):
        print(
            f"District {i+1}: Pop = {d_pop:,}, "
            f"Polsby-Popper = {polsby_popper(final_sorted[i], gdf):.4f}"
        )
    print(f"\nTotal Population: {total_pop:,}")
    print(f"Compactness Score (Σ J_d): {score:.2f}")

    # JSON output
    out = {
        "state_code": scode,
        "districts": final_sorted,
        "score": score,
        "total_population": total_pop,
        "ideal_population": ideal_pop,
        "final_population_counts": final_pop_counts,
    }
    out_path = Path.cwd() / f"districts_{scode}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
