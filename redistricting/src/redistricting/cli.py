from __future__ import annotations
level=logging.DEBUG if debug else logging.INFO,
format="%(asctime)s - %(levelname)s - %(message)s",
handlers=[logging.StreamHandler(sys.stdout)],
)




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
logging.info(f"Starting redistricting for {st.name} with {D} districts. Ideal pop: {ideal_pop:.2f}")


initial = initial_assignment(
gdf, G, D, ideal_pop,
pop_tolerance_ratio=settings.defaults.pop_tolerance_ratio,
)


final, score = optimize_districts(
initial, gdf, G, ideal_pop,
pop_tolerance_ratio=settings.defaults.pop_tolerance_ratio,
compactness_threshold=settings.defaults.compactness_threshold,
)


final_sorted = [sorted(list(d)) for d in final]
final_sorted.sort()


plot_districts(gdf, final_sorted, st.name, scode)


final_pop_counts = [sum(G.nodes[b]['pop'] for b in d) for d in final_sorted]
print("\n" + "="*50)
print(f"Final District Map for {st.name} ({D} districts)")
print("="*50)
for i, d_pop in enumerate(final_pop_counts):
print(f"District {i+1}: Pop = {d_pop:,}, Polsby-Popper = {polsby_popper(final_sorted[i], gdf):.4f}")
print(f"\nTotal Population: {total_pop:,}")
print(f"Compactness Score (Σ J_d): {score:.2f}")


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


# cleanup tempdir automatically when 'paths' goes out of scope


if __name__ == "__main__":
main()
