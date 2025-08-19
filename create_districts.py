import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree
import hashlib
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
    Returns paths to the shapefile, population file, and geographic header file.
    """
    logging.info("Step 1 of 5: Unzipping data files...")
    shapefile_zip = os.path.join(base_dir, f"tl_2024_{fips}_tabblock20.zip")
    popfile_zip = os.path.join(base_dir, f"{state_code.lower()}2020.pl.zip")

    if not os.path.exists(shapefile_zip):
        raise FileNotFoundError(f"Shapefile zip not found at {shapefile_zip}")
    if not os.path.exists(popfile_zip):
        raise FileNotFoundError(f"Population file zip not found at {popfile_zip}")

    try:
        with zipfile.ZipFile(shapefile_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        with zipfile.ZipFile(popfile_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
    except zipfile.BadZipFile as e:
        raise zipfile.BadZipFile(f"Could not unzip files. Error: {e}")

    # Find files
    shapefile = None
    pop_file = None
    geo_file = None
    for root, _, files in os.walk(temp_dir):
        for f in files:
            if f.endswith('.shp') and f.startswith(f'tl_2024_{fips}_tabblock'):
                shapefile = os.path.join(root, f)
            if f == f'{state_code.lower()}000012020.pl':
                pop_file = os.path.join(root, f)
            if f == f'{state_code.lower()}geo2020.pl':
                geo_file = os.path.join(root, f)

    if not all([shapefile, pop_file, geo_file]):
        raise FileNotFoundError("Required files not found in unzipped data.")

    logging.info("Unzipping complete.")
    return shapefile, pop_file, geo_file


def build_adjacency_graph(gdf):
    """
    Builds adjacency graph using an STRtree spatial index for efficiency.
    Avoids O(n^2) pairwise checks by querying only nearby geometries.
    """
    logging.info("Building adjacency graph with spatial index...")

    G = nx.Graph()

    # Add nodes first
    for _, row in gdf.iterrows():
        G.add_node(
            row['GEOID20'],
            pop=row['P1_001N'],
            x=row['x'],
            y=row['y'],
            geom=row.geometry,
        )

    # Build spatial index
    geoms = gdf.geometry.values
    tree = STRtree(geoms)
    geom_to_id = dict(zip(geoms, gdf['GEOID20']))

    # Loop through geometries and add edges only for touching neighbors
    for idx, geom in enumerate(geoms):
        if idx % 5000 == 0:
            logging.info(f"Processed {idx}/{len(geoms)} geometries for adjacency...")

        possible_neighbors = tree.query(geom)
        for nbr in possible_neighbors:
            if geom == nbr:
                continue
            if geom.touches(nbr):
                G.add_edge(geom_to_id[geom], geom_to_id[nbr])

    logging.info(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G


def load_and_preprocess_data(shapefile_path, pop_file_path, geo_file_path):
    """
    Loads shapefile and population data, merges them, and builds a graph.
    Returns GeoDataFrame, networkx Graph, and total state population.
    """
    logging.info("Step 2 of 5: Loading and preprocessing data...")

    # Load and reproject shapefile
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.to_crs(epsg=26915)
    gdf['centroid'] = gdf.geometry.centroid
    gdf['x'] = gdf.centroid.x
    gdf['y'] = gdf.centroid.y

    # Load geo and population files
    geo_df = pd.read_csv(geo_file_path, sep='|', header=None, dtype=str)
    pop_df = pd.read_csv(pop_file_path, sep='|', header=None, dtype=str)

    geo_df_filtered = geo_df[geo_df.iloc[:, 2] == '750'].copy()
    geo_df_filtered = geo_df_filtered.iloc[:, [7, 8]]
    geo_df_filtered.columns = ['LOGRECNO', 'GEOID']

    pop_df = pop_df[[4, 5]]
    pop_df.columns = ['LOGRECNO', 'P1_001N']

    merged_df = geo_df_filtered.merge(pop_df, on='LOGRECNO', how='inner')
    merged_df['GEOID'] = merged_df['GEOID'].str.replace('7500000US', '', regex=False)

    gdf = gdf.merge(merged_df, left_on='GEOID20', right_on='GEOID', how='left')

    if gdf['P1_001N'].isna().any():
        raise ValueError("Missing population data for some blocks.")

    gdf['P1_001N'] = gdf['P1_001N'].astype(int)
    total_pop = gdf['P1_001N'].sum()

    # Build adjacency graph with spatial index
    G = build_adjacency_graph(gdf)

    return gdf, G, total_pop
