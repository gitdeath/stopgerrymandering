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

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Redistricting algorithm using local zipped files")
parser.add_argument("--state", required=True, help="Two-letter state code (e.g., MO, CA)")
parser.add_argument("--districts", type=int, help="Number of districts (default: congressional)")
args = parser.parse_args()

# Validate state code
state_code = args.state.upper()
if state_code not in STATE_METADATA:
    raise ValueError(f"Invalid state code: {state_code}. Use two-letter code (e.g., MO, CA).")

fips, state_name, default_districts = STATE_METADATA[state_code]
D = args.districts if args.districts else default_districts

# Unzip files and find shapefile and population file
def unzip_and_find_files(base_dir, fips, state_code, temp_dir):
    shapefile_zip = os.path.join(base_dir, f"tl_2024_{fips}_tabblock20.zip")
    popfile_zip = os.path.join(base_dir, f"{state_code.lower()}2020.pl.zip")
    
    print(f"Checking for shapefile zip: {shapefile_zip}")
    print(f"Checking for population zip: {popfile_zip}")
    
    if not os.path.exists(shapefile_zip):
        raise FileNotFoundError(f"Shapefile zip not found at {shapefile_zip}")
    if not os.path.exists(popfile_zip):
        raise FileNotFoundError(f"Population file zip not found at {popfile_zip}")
    
    # Unzip shapefile
    print(f"Unzipping shapefile to {temp_dir}")
    with zipfile.ZipFile(shapefile_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Unzip population file
    print(f"Unzipping population file to {temp_dir}")
    with zipfile.ZipFile(popfile_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Find shapefile and population file
    shapefile = None
    pop_file = None
    geo_file = None
    print(f"Searching for files in {temp_dir}")
    for root, dirs, files in os.walk(temp_dir):
        print(f"Directory: {root}")
        print(f"Subdirectories: {dirs}")
        print(f"Files: {files}")
        for f in files:
            if f.endswith('.shp') and f.startswith(f'tl_2024_{fips}_tabblock'):
                shapefile = os.path.join(root, f)
            if f == f'{state_code.lower()}000012020.pl':
                pop_file = os.path.join(root, f)
            if f == f'{state_code.lower()}geo2020.pl':
                geo_file = os.path.join(root, f)
    
    if not shapefile:
        raise FileNotFoundError(f"Shapefile not found in unzipped data (expected 'tl_2024_{fips}_tabblock*.shp')")
    if not pop_file:
        raise FileNotFoundError(f"Population file not found in unzipped data (expected '{state_code.lower()}000012020.pl')")
    if not geo_file:
        raise FileNotFoundError(f"Geographic header file not found in unzipped data (expected '{state_code.lower()}geo2020.pl')")
    
    print(f"Found shapefile: {shapefile}")
    print(f"Found population file: {pop_file}")
    print(f"Found geographic header file: {geo_file}")
    return shapefile, pop_file, geo_file

# Load and preprocess data
def load_data(shapefile_path, pop_file_path, geo_file_path):
    print(f"Loading shapefile: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    print(f"Shapefile loaded. Shape: {gdf.shape}")
    # Reproject to NAD83 / Missouri Central (EPSG:26915)
    print("Reprojecting shapefile to EPSG:26915")
    gdf = gdf.to_crs(epsg=26915)
    gdf['centroid'] = gdf.geometry.centroid
    gdf['x'] = gdf.centroid.x
    gdf['y'] = gdf.centroid.y
    
    print(f"Loading population data from {pop_file_path} and {geo_file_path}")
    
    # Read the geographic header file (pipe-delimited)
    geo_df = pd.read_csv(geo_file_path, sep='|', header=None, dtype=str)
    print(f"Geographic file loaded. Shape: {geo_df.shape}")
    
    # Filter for Census Blocks using the correct SUMLEVEL code '750'
    geo_df_filtered = geo_df[geo_df.iloc[:, 2] == '750'].copy()
    print(f"Geographic file filtered for SUMLEVEL '750'. New shape: {geo_df_filtered.shape}")

    if geo_df_filtered.empty:
      raise ValueError("Filtered geographic data is empty. The provided file does not contain Census Block data with SUMLEVEL '750'.")

    # Select the required columns: `LOGRECNO` (index 7) and `GEOID` (index 8).
    geo_df_filtered = geo_df_filtered.iloc[:, [7, 8]]
    geo_df_filtered.columns = ['LOGRECNO', 'GEOID']

    # Read the population data file (pipe-delimited)
    pop_df = pd.read_csv(pop_file_path, sep='|', header=None, dtype=str)
    print(f"Population file loaded. Shape: {pop_df.shape}")
    
    # Select the required columns: `LOGRECNO` (index 4) and `P1_001N` (index 5).
    pop_df = pop_df[[4, 5]]
    pop_df.columns = ['LOGRECNO', 'P1_001N']
    
    # Merge the geographic and population dataframes on LOGRECNO
    merged_df = geo_df_filtered.merge(pop_df, on='LOGRECNO', how='inner')
    print(f"Geographic and Population data merged. New shape: {merged_df.shape}")
    
    # FIX: Remove the '7500000US' prefix to match the shapefile's GEOID20 format
    merged_df['GEOID'] = merged_df['GEOID'].str.replace('7500000US', '', regex=False)
    
    print(f"Population file GEOID sample: {merged_df['GEOID'].head().tolist()}")
    print(f"Shapefile GEOID20 sample: {gdf['GEOID20'].head().tolist()}")
    
    # Merge with the GeoDataFrame
    gdf = gdf.merge(merged_df, left_on='GEOID20', right_on='GEOID', how='left')
    print(f"Final GeoDataFrame merged with population data. New shape: {gdf.shape}")
    
    if gdf['P1_001N'].isna().any():
        unmatched = gdf[gdf['P1_001N'].isna()]['GEOID20'].tolist()
        print(f"Unmatched GEOID20 values (first 10): {unmatched[:10]}")
        raise ValueError(f"Missing population data for {len(unmatched)} blocks")
    
    # The rest of the function remains the same, using the new gdf
    G = nx.Graph()
    for idx1, row1 in gdf.iterrows():
        G.add_node(row1['GEOID20'], pop=row1['P1_001N'], x=row1['x'], y=row1['y'], geom=row1.geometry)
        for idx2, row2 in gdf.iterrows():
            if idx1 < idx2 and row1.geometry.touches(row2.geometry):
                G.add_edge(row1['GEOID20'], row2['GEOID20'])
    
    total_pop = gdf['P1_001N'].astype(int).sum()
    return gdf, G, total_pop

# Determine sweep order
def get_sweep_order(gdf):
    block_ids = sorted(gdf['GEOID20'])
    hash_input = ''.join(block_ids).encode()
    hash_val = hashlib.sha256(hash_input).hexdigest()
    sweep_idx = int(hash_val, 16) % 4
    
    if sweep_idx == 0:  # NE: descending y, ascending x
        return gdf.sort_values(by=['y', 'x'], ascending=[False, True])['GEOID20'].tolist()
    elif sweep_idx == 1:  # SW: ascending y, ascending x
        return gdf.sort_values(by=['y', 'x'], ascending=[True, True])['GEOID20'].tolist()
    elif sweep_idx == 2:  # SE: ascending y, descending x
        return gdf.sort_values(by=['y', 'x'], ascending=[True, False])['GEOID20'].tolist()
    else:  # NW: descending y, descending x
        return gdf.sort_values(by=['y', 'x'], ascending=[False, False])['GEOID20'].tolist()

# Compute moment of inertia
def compute_inertia(district_blocks, gdf):
    district_gdf = gdf[gdf['GEOID20'].isin(district_blocks)]
    total_pop = district_gdf['P1_001N'].astype(int).sum()
    if total_pop == 0:
        return float('inf')
    centroid_x = (district_gdf['P1_001N'].astype(int) * district_gdf['x']).sum() / total_pop
    centroid_y = (district_gdf['P1_001N'].astype(int) * district_gdf['y']).sum() / total_pop
    inertia = ((district_gdf['P1_001N'].astype(int) * ((district_gdf['x'] - centroid_x)**2 + (district_gdf['y'] - centroid_y)**2)).sum())
    return inertia

# Compute Polsby-Popper score
def polsby_popper(district_blocks, gdf):
    district_gdf = gdf[gdf['GEOID20'].isin(district_blocks)]
    union_geom = district_gdf.geometry.unary_union
    if isinstance(union_geom, Polygon):
        area = union_geom.area
        perimeter = union_geom.length
        return (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    return 0

# Check contiguity
def is_contiguous(district_blocks, G):
    subgraph = G.subgraph(district_blocks)
    return nx.is_connected(subgraph)

# Initial district assignment
def initial_assignment(gdf, G, D, ideal_pop):
    sweep_order = get_sweep_order(gdf)
    districts = [set() for _ in range(D)]
    pop_per_district = [0] * D
    pop_tolerance = ideal_pop * 0.005
    
    for block in sweep_order:
        block_pop = int(G.nodes[block]['pop'])
        min_pop_idx = min(range(D), key=lambda i: pop_per_district[i])
        if (pop_per_district[min_pop_idx] + block_pop <= ideal_pop + pop_tolerance and
                (not districts[min_pop_idx] or any(G.has_edge(block, b) for b in districts[min_pop_idx]))):
            districts[min_pop_idx].add(block)
            pop_per_district[min_pop_idx] += block_pop
    
    return districts

# Optimization objective
def objective(districts, gdf, G, ideal_pop):
    total_inertia = 0
    pop_tolerance = ideal_pop * 0.005
    for d in districts:
        district_pop = sum(int(G.nodes[b]['pop']) for b in d)
        if not (ideal_pop - pop_tolerance <= district_pop <= ideal_pop + pop_tolerance):
            return float('inf')
        if not is_contiguous(d, G):
            return float('inf')
        if polsby_popper(d, gdf) < 0.20:
            return float('inf')
        total_inertia += compute_inertia(d, gdf)
    return total_inertia

# Simulated annealing optimization
def optimize_districts(districts, gdf, G, ideal_pop, max_iter=1000):
    def perturb(districts):
        new_districts = [set(d) for d in districts]
        d1, d2 = np.random.choice(len(districts), 2, replace=False)
        if new_districts[d1] and new_districts[d2]:
            block = np.random.choice(list(new_districts[d1]))
            if any(G.has_edge(block, b) for b in new_districts[d2]):
                new_districts[d1].remove(block)
                new_districts[d2].add(block)
        return new_districts
    
    current_districts = districts
    current_score = objective(districts, gdf, G, ideal_pop)
    temp, alpha, max_iter = 1000.0, 0.95, max_iter
    
    for i in range(max_iter):
        new_districts = perturb(current_districts)
        new_score = objective(new_districts, gdf, G, ideal_pop)
        if new_score < current_score or np.random.rand() < np.exp((current_score - new_score) / temp):
            current_districts = new_districts
            current_score = new_score
        temp *= alpha
    
    return current_districts, current_score

# Tie-breaker
def apply_tie_breaker(district_sets, gdf):
    sorted_districts = [sorted(d) for d in district_sets]
    sorted_districts.sort()
    return sorted_districts

# Main function
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = tempfile.mkdtemp(prefix=f"redistrict_{state_code}_")
    print(f"Created temporary directory: {temp_dir}")
    try:
        # Unzip files and find paths
        shapefile_path, pop_file_path, geo_file_path = unzip_and_find_files(base_dir, fips, state_code, temp_dir)
        
        # Load data and compute total population
        gdf, G, total_pop = load_data(shapefile_path, pop_file_path, geo_file_path)
        ideal_pop = total_pop / D
        
        # Initial assignment
        districts = initial_assignment(gdf, G, D, ideal_pop)
        
        # Optimize
        final_districts, final_score = optimize_districts(districts, gdf, G, ideal_pop)
        
        # Apply tie-breaker
        final_districts = apply_tie_breaker(final_districts, gdf)
        
        # Output
        print(f"District Map for {state_name} ({D} districts):", [[b for b in d] for d in final_districts])
        print(f"Total Population: {total_pop}")
        print(f"Compactness Score (Σ J_d): {final_score}")
        
        # Save to JSON
        import json
        with open(f"districts_{state_code}.json", "w") as f:
            json.dump({"districts": [[b for b in d] for d in final_districts], "score": final_score}, f)
    
    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"Temporary directory {temp_dir} was not deleted for inspection.")
    
    print(f"Check {temp_dir} for unzipped files before exiting.")

if __name__ == "__main__":
    main()
