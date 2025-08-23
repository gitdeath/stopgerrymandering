from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Settings
from .io import unzip_and_find_files, load_and_preprocess_data
from .assign import initial_assignment
from .optimize import fix_contiguity, powerful_balancer, perfect_map
from .metrics import polsby_popper, is_contiguous, compute_inertia
from .viz import plot_districts


def parse_args():
    # This function is unchanged
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
        help="Enable detailed debugging output and save intermediate maps.",
    )
    return parser.parse_args()


def setup_logging(debug: bool):
    # This function is unchanged
    log_filename = "redistricting_run.log"
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    logging.info(f"Detailed log being saved to: {log_filename}")


def print_debug_stats(phase_name: str, districts, gdf, G):
    """Helper function to print district stats during debug runs."""
    header = f" DEBUG STATS: After {phase_name} "
    logging.info("\n" + header.center(80, "-"))
    
    for i, d in enumerate(districts):
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


def main():
    args = parse_args()
    setup_logging(args.debug)
    settings = Settings.load(args.config)
    scode = args.state.upper()
    if scode not in settings.states:
        raise ValueError(f"Invalid state code: {scode}. Use a two-letter code.")
    st = settings.states[scode]
    D = args.districts if args.districts else st.districts
    base_dir = Path(__file__).resolve().parents[2]
    paths = unzip_and_find_files(base_dir, st.fips, scode)
    gdf, G, total_pop = load_and_preprocess_data(paths, settings.defaults.crs_epsg)
    ideal_pop = total_pop / D
    logging.info(
        f"Starting redistricting for {st.name} with {D} districts. "
        f"Ideal pop: {ideal_pop:.2f}"
    )
    
    logging.info("Starting Stage 0: Initial Assignment...")
    initial = initial_assignment(
        gdf, G, D, ideal_pop,
        pop_tolerance_ratio=settings.defaults.pop_tolerance_ratio,
    )
    if args.debug:
        plot_districts(gdf, initial, st.name, scode, output_filename=f"debug_1_initial_{scode}.png")
        print_debug_stats("Initial Assignment", initial, gdf, G)

    logging.info("Starting Stage 1: Contiguity Repair...")
    contiguous_map = fix_contiguity(initial, gdf, G)
    if args.debug:
        plot_districts(gdf, contiguous_map, st.name, scode, output_filename=f"debug_2_contiguous_{scode}.png")
        print_debug_stats("Contiguity Repair", contiguous_map, gdf, G)

    logging.info("Starting Stage 2: Powerful Balancing...")
    balanced_map = powerful_balancer(
        contiguous_map, gdf, G, ideal_pop,
        pop_tolerance_ratio=settings.defaults.pop_tolerance_ratio,
    )
    if args.debug:
        plot_districts(gdf, balanced_map, st.name, scode, output_filename=f"debug_3_balanced_{scode}.png")
        print_debug_stats("Powerful Balancing", balanced_map, gdf, G)

    logging.info("Starting Stage 3: Final Perfecting...")
    # --- THE FIX ---
    # Update the function call to pass the new arguments needed for the intermediate debug step
    final_map, final_score = perfect_map(
        balanced_map, gdf, G, ideal_pop,
        pop_tolerance_ratio=settings.defaults.pop_tolerance_ratio,
        st=st,
        scode=scode,
        debug=args.debug
    )
    # --- END FIX ---
    
    final_sorted = [sorted(list(d)) for d in final_map]
    final_sorted.sort()
    plot_districts(gdf, final_sorted, st.name, scode, output_filename=f"districts_{scode}.png")
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
    print(f"Final Score: {final_score:.2f}")
    
    out = {
        "state_code": scode, "districts": final_sorted, "score": final_score,
        "total_population": total_pop, "ideal_population": ideal_pop,
        "final_population_counts": final_pop_counts,
    }
    out_path = Path.cwd() / f"districts_{scode}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
