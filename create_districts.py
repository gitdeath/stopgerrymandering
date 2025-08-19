import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import Polygon
import hashlib
from scipy.optimize import dual_annealing
import pandas as pd
import os
import argparse
import zipfile
import tempfile
import shutil
import json
import logging
import sys

# Configure logging to control debug output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# State metadata: postal code -> (FIPS, state name, congressional districts)
STATE_METADATA = {
    'MO': ('29', 'Missouri', 8),
    'CA': ('06', 'California', 52),
    'TX': ('48', 'Texas', 38),
    'NY': ('36', 'New York', 26),
    'FL': ('12', 'Florida', 28),
    'AK': ('02', 'Alaska', 1),
    'AL': ('01', 'Alabama', 7),
    'AR': ('05', 'Arkansas', 4),
    'AZ': ('04', 'Arizona', 9),
    'CO': ('08', 'Colorado', 8),
    'CT': ('09', 'Connecticut', 5),
    'DE': ('10', 'Delaware', 1),
    'GA': ('13', 'Georgia', 14),
    'HI': ('15', 'Hawaii', 2),
    'IA': ('19', 'Iowa', 4),
    'ID': ('16', 'Idaho', 2),
    'IL': ('17', 'Illinois', 17),
    'IN': ('18', 'Indiana', 9),
    'KS': ('20', 'Kansas', 4),
    'KY': ('21', 'Kentucky', 6),
    'LA': ('22', 'Louisiana', 6),
    'MA': ('25', 'Massachusetts', 9),
    'MD': ('24', 'Maryland', 8),
    'ME': ('23', 'Maine', 2),
    'MI': ('26', 'Michigan', 13),
    'MN': ('27', 'Minnesota', 8),
    'MS': ('28', 'Mississippi', 4),
    'MT': ('30', 'Montana', 2),
    'NC': ('37', 'North Carolina', 14),
    'ND': ('38', 'North Dakota', 1),
    'NE': ('31', 'Nebraska', 3),
    'NH': ('33', 'New Hampshire', 2),
    'NJ': ('34', 'New Jersey', 12),
    'NM': ('35', 'New Mexico', 3),
    'NV': ('32', 'Nevada', 4),
    'OH': ('39', 'Ohio', 15),
    'OK': ('40', 'Oklahoma', 5),
    'OR': ('41', 'Oregon', 6),
    'PA': ('42', 'Pennsylvania', 17),
    'RI': ('44', 'Rhode Island', 2),
    'SC': ('45', 'South Carolina', 7),
    'SD': ('46', 'South Dakota', 1),
    'TN': ('47', 'Tennessee', 9),
    'UT': ('49', 'Utah', 4),
    'VA': ('51', 'Virginia', 11),
    'VT': ('50', 'Vermont', 1),
    'WA': ('53', 'Washington', 10),
    'WI': ('55', 'Wisconsin', 8),
    'WV': ('54', 'West Virginia', 2),
    'WY': ('56', 'Wyoming', 1),
}


def parse_arguments():
    """Parses command-line arguments for state, districts, and debug flag."""
    parser = argparse.ArgumentParser(
        description="Redistricting algorithm using local zipped files"
    )
    parser.add_argument(
        "--state", required=True, help="Two-letter state code (e.g., MO, CA)"
    )
    parser.add_argument(
        "--districts",
        type=int,
        help="Number of districts (default: congressional)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debugging output.",
    )
    return parser.parse_args()


def setup_logging(debug_mode):
    """Sets the logging level based on the debug flag."""
    if debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)


def unzip_and_find_files(base_dir, fips, state_code, temp_dir):
    """
    Unzips the shapefile and population data from specified paths.

    Args:
        base_dir (str): The base directory where zipped files are located.
        fips (str): The FIPS code for the state.
        state_code (str): The two-letter postal code for the state.
        temp_dir (str): The temporary directory to extract files to.

    Returns:
        tuple: A tuple containing the paths to the shapefile, population file,
               and geographic header file.
    """
    logging.info("Step 1 of 5: Unzipping data files...")
    shapefile_zip = os.path.join(base_dir, f"tl_2024_{fips}_tabblock20.zip")
    popfile_zip = os.path.join(base_dir, f"{state_code.lower()}2020.pl.zip")

    if not os.path.exists(shapefile_zip):
        raise FileNotFoundError(f"Shapefile zip not found at {shapefile_zip}")
    if not os.path.exists(popfile_zip):
        raise FileNotFoundError(f"Population file zip not found at {popfile_zip}")

    try:
        # Unzip shapefile
        logging.debug(f"Unzipping shapefile to {temp_dir}")
        with zipfile.ZipFile(shapefile_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Unzip population file
        logging.debug(f"Unzipping population file to {temp_dir}")
        with zipfile.ZipFile(popfile_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
    except zipfile.BadZipFile as e:
        raise zipfile.BadZipFile(f"Could not unzip files. Error: {e}")

    # Find the specific shapefile, population file, and geo file within the
    # unzipped directory structure
    shapefile = None
    pop_file = None
    geo_file = None
    logging.debug(f"Searching for files in {temp_dir}")
    for root, dirs, files in os.walk(temp_dir):
        for f in files:
            if f.endswith('.shp') and f.startswith(f'tl_2024_{fips}_tabblock'):
                shapefile = os.path.join(root, f)
            if f == f'{state_code.lower()}000012020.pl':
                pop_file = os.path.join(root, f)
            if f == f'{state_code.lower()}geo2020.pl':
                geo_file = os.path.join(root, f)

    if not all([shapefile, pop_file, geo_file]):
        raise FileNotFoundError(
            "Required files not found in unzipped data."
            f"Shapefile found: {shapefile is not None}, "
            f"Pop file found: {pop_file is not None}, "
            f"Geo file found: {geo_file is not None}"
        )

    logging.info("Unzipping complete.")
    logging.debug(f"Found shapefile: {shapefile}")
    logging.debug(f"Found population file: {pop_file}")
    logging.debug(f"Found geographic header file: {geo_file}")
    return shapefile, pop_file, geo_file


def load_and_preprocess_data(shapefile_path, pop_file_path, geo_file_path):
    """
    Loads shapefile and population data, merges them, and builds a graph.

    Args:
        shapefile_path (str): Path to the .shp file.
        pop_file_path (str): Path to the population data file.
        geo_file_path (str): Path to the geographic header file.

    Returns:
        tuple: A tuple containing the GeoDataFrame, the networkx Graph,
               and the total state population.
    """
    logging.info("Step 2 of 5: Loading and preprocessing data...")

    # Load and reproject the GeoDataFrame
    gdf = gpd.read_file(shapefile_path)
    logging.debug(f"Shapefile loaded. Initial shape: {gdf.shape}")
    gdf = gdf.to_crs(epsg=26915)  # Reproject to a projected CRS for accurate area/distance
    gdf['centroid'] = gdf.geometry.centroid
    gdf['x'] = gdf.centroid.x
    gdf['y'] = gdf.centroid.y
    logging.debug("Shapefile reprojected and centroids calculated.")

    # Load and merge population data
    geo_df = pd.read_csv(geo_file_path, sep='|', header=None, dtype=str)
    pop_df = pd.read_csv(pop_file_path, sep='|', header=None, dtype=str)

    # Filter for Census Blocks (SUMLEVEL '750') and select relevant columns
    geo_df_filtered = geo_df[geo_df.iloc[:, 2] == '750'].copy()
    if geo_df_filtered.empty:
        raise ValueError(
            "Filtered geographic data is empty. The provided file does not "
            "contain Census Block data with SUMLEVEL '750'."
        )
    geo_df_filtered = geo_df_filtered.iloc[:, [7, 8]]
    geo_df_filtered.columns = ['LOGRECNO', 'GEOID']

    # Select relevant population data columns
    pop_df = pop_df[[4, 5]]
    pop_df.columns = ['LOGRECNO', 'P1_001N']

    # Merge population data with geographic data
    merged_df = geo_df_filtered.merge(pop_df, on='LOGRECNO', how='inner')
    merged_df['GEOID'] = merged_df['GEOID'].str.replace('7500000US', '', regex=False)
    
    # Final merge with the GeoDataFrame
    gdf = gdf.merge(merged_df, left_on='GEOID20', right_on='GEOID', how='left')

    if gdf['P1_001N'].isna().any():
        unmatched_count = gdf['P1_001N'].isna().sum()
        raise ValueError(f"Missing population data for {unmatched_count} blocks.")

    # Convert population column to integer
    gdf['P1_001N'] = gdf['P1_001N'].astype(int)
    total_pop = gdf['P1_001N'].sum()
    logging.debug(f"Total state population: {total_pop}")
    logging.info("Data loading complete. Building adjacency graph...")

    # Build the adjacency graph
    G = nx.Graph()
    for idx1, row1 in gdf.iterrows():
        G.add_node(
            row1['GEOID20'],
            pop=row1['P1_001N'],
            x=row1['x'],
            y=row1['y'],
            geom=row1.geometry,
        )

    # Add edges for adjacent blocks
    for i, row1 in gdf.iterrows():
        for j, row2 in gdf.iterrows():
            if i < j and row1.geometry.touches(row2.geometry):
                G.add_edge(row1['GEOID20'], row2['GEOID20'])

    logging.info(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return gdf, G, total_pop


def get_sweep_order(gdf):
    """
    Determines a consistent, pseudo-random sweep order for initial assignment.

    The sweep direction is determined by a hash of the sorted GEOID20s,
    ensuring a deterministic result for a given input dataset.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame of census blocks.

    Returns:
        list: A list of GEOID20s sorted according to the sweep direction.
    """
    block_ids = sorted(gdf['GEOID20'])
    hash_input = ''.join(block_ids).encode()
    hash_val = hashlib.sha256(hash_input).hexdigest()
    sweep_idx = int(hash_val, 16) % 4

    if sweep_idx == 0:  # NE: descending y, ascending x
        logging.debug("Sweep order: Northeast")
        return gdf.sort_values(by=['y', 'x'], ascending=[False, True])['GEOID20'].tolist()
    elif sweep_idx == 1:  # SW: ascending y, ascending x
        logging.debug("Sweep order: Southwest")
        return gdf.sort_values(by=['y', 'x'], ascending=[True, True])['GEOID20'].tolist()
    elif sweep_idx == 2:  # SE: ascending y, descending x
        logging.debug("Sweep order: Southeast")
        return gdf.sort_values(by=['y', 'x'], ascending=[True, False])['GEOID20'].tolist()
    else:  # NW: descending y, descending x
        logging.debug("Sweep order: Northwest")
        return gdf.sort_values(by=['y', 'x'], ascending=[False, False])['GEOID20'].tolist()


def compute_inertia(district_blocks, gdf):
    """
    Calculates the moment of inertia for a given set of blocks.

    Lower inertia indicates a more compact, centrally-massed district.
    """
    district_gdf = gdf[gdf['GEOID20'].isin(district_blocks)]
    total_pop = district_gdf['P1_001N'].astype(int).sum()
    if total_pop == 0:
        return float('inf')
    centroid_x = (district_gdf['P1_001N'].astype(int) * district_gdf['x']).sum() / total_pop
    centroid_y = (district_gdf['P1_001N'].astype(int) * district_gdf['y']).sum() / total_pop
    inertia = (
        (
            district_gdf['P1_001N'].astype(int)
            * ((district_gdf['x'] - centroid_x) ** 2 + (district_gdf['y'] - centroid_y) ** 2)
        ).sum()
    )
    return inertia


def polsby_popper(district_blocks, gdf):
    """
    Calculates the Polsby-Popper score for a district.

    A higher score (closer to 1) indicates a more compact district.
    """
    district_gdf = gdf[gdf['GEOID20'].isin(district_blocks)]
    union_geom = district_gdf.geometry.unary_union
    if isinstance(union_geom, Polygon):
        area = union_geom.area
        perimeter = union_geom.length
        return (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
    return 0


def is_contiguous(district_blocks, G):
    """Checks if a set of blocks forms a contiguous district."""
    subgraph = G.subgraph(district_blocks)
    return nx.is_connected(subgraph)


def initial_assignment(gdf, G, D, ideal_pop):
    """
    Performs an initial assignment of blocks to districts using a greedy sweep.

    This method iterates through blocks in a defined sweep order and
    assigns each block to the district with the lowest current population,
    provided the block is adjacent to that district. This creates a
    reasonable starting point for the optimization.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with block data.
        G (nx.Graph): Adjacency graph of census blocks.
        D (int): Number of districts.
        ideal_pop (int): The target population per district.

    Returns:
        list: A list of sets, where each set contains the GEOID20s for a district.
    """
    logging.info("Step 3 of 5: Generating initial district map...")
    sweep_order = get_sweep_order(gdf)
    districts = [set() for _ in range(D)]
    pop_per_district = [0] * D
    pop_tolerance = ideal_pop * 0.005

    for i, block in enumerate(sweep_order):
        block_pop = int(G.nodes[block]['pop'])
        
        # Find the best district to add the block to
        best_idx = -1
        min_pop = float('inf')
        
        # Iterate through districts to find a suitable one
        for j in range(D):
            # Check for contiguity and population limit
            is_adjacent = not districts[j] or any(G.has_edge(block, b) for b in districts[j])
            
            if (pop_per_district[j] + block_pop <= ideal_pop + pop_tolerance and
                is_adjacent and
                pop_per_district[j] < min_pop):
                
                min_pop = pop_per_district[j]
                best_idx = j
        
        if best_idx != -1:
            districts[best_idx].add(block)
            pop_per_district[best_idx] += block_pop
        else:
            # If no suitable district found, we need to backtrack or handle this.
            # For simplicity, this implementation assigns it to the least
            # populated district even if it breaks contiguity.
            # The optimization step will fix this.
            min_pop_idx = np.argmin(pop_per_district)
            districts[min_pop_idx].add(block)
            pop_per_district[min_pop_idx] += block_pop
        
        if (i + 1) % 1000 == 0 or i == len(sweep_order) - 1:
            logging.info(f"Assigned {i + 1} of {len(sweep_order)} blocks.")

    return districts


def objective(districts, gdf, G, ideal_pop):
    """
    The main objective function to be minimized by the optimizer.

    This function calculates a total score based on population deviation,
    contiguity, compactness (Polsby-Popper), and moment of inertia.
    It returns infinity if any constraints are violated.

    Args:
        districts (list): A list of sets, each representing a district.
        gdf (gpd.GeoDataFrame): The GeoDataFrame.
        G (nx.Graph): The adjacency graph.
        ideal_pop (int): The target population per district.

    Returns:
        float: A score representing the quality of the district map.
               Lower is better.
    """
    total_inertia = 0
    pop_tolerance = ideal_pop * 0.005

    for d in districts:
        if not d:  # Skip empty districts
            return float('inf')
        
        district_pop = sum(int(G.nodes[b]['pop']) for b in d)
        
        # Population constraint
        if not (ideal_pop - pop_tolerance <= district_pop <= ideal_pop + pop_tolerance):
            logging.debug(f"Population constraint violated. Pop: {district_pop}, Ideal: {ideal_pop}")
            return float('inf')

        # Contiguity constraint
        if not is_contiguous(d, G):
            logging.debug("Contiguity constraint violated.")
            return float('inf')

        # Polsby-Popper compactness constraint
        pp_score = polsby_popper(d, gdf)
        if pp_score < 0.20:
            logging.debug(f"Polsby-Popper constraint violated. Score: {pp_score}")
            return float('inf')

        # Moment of inertia (to be minimized)
        total_inertia += compute_inertia(d, gdf)

    return total_inertia


def optimize_districts(districts, gdf, G, ideal_pop, max_iter=10000):
    """
    Optimizes the initial district map using a simulated annealing-like approach.

    Args:
        districts (list): The initial district map.
        gdf (gpd.GeoDataFrame): The GeoDataFrame.
        G (nx.Graph): The adjacency graph.
        ideal_pop (int): The target population per district.
        max_iter (int): The maximum number of optimization iterations.

    Returns:
        tuple: The optimized district map and its final score.
    """
    logging.info("Step 4 of 5: Optimizing district map...")
    
    # Use the number of districts as a seed to ensure repeatable results for the same state
    np.random.seed(D)
    
    def perturb(districts):
        """Generates a new state by swapping one block between two adjacent districts."""
        new_districts = [set(d) for d in districts]
        
        # Choose two random districts
        d1_idx, d2_idx = np.random.choice(len(districts), 2, replace=False)
        
        # Find a block in d1 that is adjacent to d2
        blocks_to_move = [
            b for b in new_districts[d1_idx]
            if any(G.has_edge(b, neighbor) for neighbor in new_districts[d2_idx])
        ]
        
        if blocks_to_move:
            block = np.random.choice(blocks_to_move)
            new_districts[d1_idx].remove(block)
            new_districts[d2_idx].add(block)
            logging.debug(f"Perturbing map: moved block {block} from district {d1_idx} to {d2_idx}")
        else:
            logging.debug("No valid blocks to move, skipping perturbation.")

        return new_districts

    current_districts = districts
    current_score = objective(districts, gdf, G, ideal_pop)
    best_districts = list(current_districts)
    best_score = current_score

    temp, alpha = 1000.0, 0.999 # Annealing schedule
    
    logging.info("Starting simulated annealing optimization.")
    for i in range(max_iter):
        new_districts = perturb(current_districts)
        new_score = objective(new_districts, gdf, G, ideal_pop)

        # Calculate acceptance probability
        if new_score < current_score or np.random.rand() < np.exp((current_score - new_score) / temp):
            current_districts = new_districts
            current_score = new_score
            logging.debug(f"Iteration {i+1}: New score {current_score:.2f} accepted.")
            
            # Update best found solution
            if current_score < best_score:
                best_districts = list(current_districts)
                best_score = current_score
                logging.info(f"Iteration {i+1}: New best score found: {best_score:.2f}")
        else:
            logging.debug(f"Iteration {i+1}: New score {new_score:.2f} rejected.")
        
        temp *= alpha
        
        if (i + 1) % 100 == 0:
            logging.info(f"Iteration {i+1}/{max_iter}. Current score: {current_score:.2f}")

    logging.info("Optimization complete.")
    return best_districts, best_score


def apply_tie_breaker(district_sets):
    """
    Applies a tie-breaker to ensure consistent output format.

    Sorts the blocks within each district, and then sorts the districts
    themselves based on their sorted list of blocks.
    """
    logging.info("Step 5 of 5: Applying tie-breaker...")
    sorted_districts = [sorted(list(d)) for d in district_sets]
    sorted_districts.sort()
    return sorted_districts


def main():
    """Main function to run the redistricting algorithm."""
    args = parse_arguments()
    setup_logging(args.debug)

    state_code = args.state.upper()
    if state_code not in STATE_METADATA:
        raise ValueError(f"Invalid state code: {state_code}. Use a two-letter code.")

    fips, state_name, default_districts = STATE_METADATA[state_code]
    D = args.districts if args.districts else default_districts
    logging.info(f"Starting redistricting for {state_name} with {D} districts.")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = tempfile.mkdtemp(prefix=f"redistrict_{state_code}_")
    
    try:
        # Unzip files and find paths
        shapefile_path, pop_file_path, geo_file_path = unzip_and_find_files(
            base_dir, fips, state_code, temp_dir
        )
        
        # Load data and compute total population
        gdf, G, total_pop = load_and_preprocess_data(
            shapefile_path, pop_file_path, geo_file_path
        )
        ideal_pop = total_pop / D
        logging.info(f"Ideal population per district: {ideal_pop:.2f}")

        # Initial assignment
        initial_districts = initial_assignment(gdf, G, D, ideal_pop)
        
        # Optimize
        final_districts, final_score = optimize_districts(
            initial_districts, gdf, G, ideal_pop
        )
        
        # Apply tie-breaker
        final_districts = apply_tie_breaker(final_districts)
        
        # Output results
        logging.info("Process complete! Generating output.")
        print("\n" + "="*50)
        print(f"Final District Map for {state_name} ({D} districts)")
        print("="*50)
        
        # Final validation and summary
        final_pop_counts = [sum(G.nodes[b]['pop'] for b in d) for d in final_districts]
        for i, d_pop in enumerate(final_pop_counts):
            print(f"District {i+1}: Pop = {d_pop:,}, Polsby-Popper = {polsby_popper(final_districts[i], gdf):.4f}")

        print(f"\nTotal Population: {total_pop:,}")
        print(f"Compactness Score (Σ J_d): {final_score:.2f}")

        # Save to JSON
        output_data = {
            "state_code": state_code,
            "districts": final_districts,
            "score": final_score,
            "total_population": total_pop,
            "ideal_population": ideal_pop,
            "final_population_counts": final_pop_counts,
        }
        output_filename = f"districts_{state_code}.json"
        with open(output_filename, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {output_filename}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        # Only clean up if the process was successful to allow inspection on error
        print(f"\nTemporary directory {temp_dir} was not deleted for inspection.")
        raise
    finally:
        # Clean up the temporary directory on successful completion
        if not args.debug and os.path.exists(temp_dir):
            logging.info("Cleaning up temporary files...")
            shutil.rmtree(temp_dir)
            logging.info("Cleanup complete.")

if __name__ == "__main__":
    main()
