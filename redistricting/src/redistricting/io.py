from __future__ import annotations
import os, zipfile, tempfile, logging
from pathlib import Path
import pandas as pd
import geopandas as gpd


from .graph import build_adjacency_graph


class DataPaths:
def __init__(self, shapefile: Path, pop_file: Path, geo_file: Path, tempdir: tempfile.TemporaryDirectory):
self.shapefile = Path(shapefile)
self.pop_file = Path(pop_file)
self.geo_file = Path(geo_file)
self.tempdir = tempdir




def unzip_and_find_files(base_dir: Path, fips: str, state_code: str) -> DataPaths:
logging.info("Step 1 of 5: Unzipping data files...")
base_dir = Path(base_dir)
shapefile_zip = base_dir / f"tl_2024_{fips}_tabblock20.zip"
popfile_zip = base_dir / f"{state_code.lower()}2020.pl.zip"


if not shapefile_zip.exists():
raise FileNotFoundError(f"Shapefile zip not found at {shapefile_zip}")
if not popfile_zip.exists():
raise FileNotFoundError(f"Population file zip not found at {popfile_zip}")


tempdir = tempfile.TemporaryDirectory(prefix=f"redistrict_{state_code}_")
with zipfile.ZipFile(shapefile_zip, 'r') as z: z.extractall(tempdir.name)
with zipfile.ZipFile(popfile_zip, 'r') as z: z.extractall(tempdir.name)


shapefile = pop = geo = None
for root, _, files in os.walk(tempdir.name):
for f in files:
if f.endswith('.shp') and f.startswith(f'tl_2024_{fips}_tabblock'):
shapefile = Path(root) / f
if f == f'{state_code.lower()}000012020.pl':
pop = Path(root) / f
if f == f'{state_code.lower()}geo2020.pl':
geo = Path(root) / f
if not all([shapefile, pop, geo]):
raise FileNotFoundError("Required files not found in unzipped data.")
logging.info("Unzipping complete.")
return DataPaths(shapefile, pop, geo, tempdir)




def load_and_preprocess_data(paths: DataPaths, crs_epsg: int):
logging.info("Step 2 of 5: Loading and preprocessing data...")
gdf = gpd.read_file(paths.shapefile)
gdf = gdf.to_crs(epsg=crs_epsg)
gdf['centroid'] = gdf.geometry.centroid
gdf['x'] = gdf.centroid.x
gdf['y'] = gdf.centroid.y


geo_df = pd.read_csv(paths.geo_file, sep='|', header=None, dtype=str)
pop_df = pd.read_csv(paths.pop_file, sep='|', header=None, dtype=str)
geo_df_filtered = geo_df[geo_df.iloc[:, 2] == '750'].copy()
if geo_df_filtered.empty:
raise ValueError("No Census Block data (SUMLEVEL '750') in geo header file.")


geo_df_filtered = geo_df_filtered.iloc[:, [7, 8]]
geo_df_filtered.columns = ['LOGRECNO', 'GEOID']


pop_df = pop_df[[4, 5]]
pop_df.columns = ['LOGRECNO', 'P1_001N']


merged_df = geo_df_filtered.merge(pop_df, on='LOGRECNO', how='inner')
merged_df['GEOID'] = merged_df['GEOID'].str.replace('7500000US', '', regex=False)


gdf = gdf.merge(merged_df, left_on='GEOID20', right_on='GEOID', how='left')
if gdf['P1_001N'].isna().any():
raise ValueError(f"Missing population data for {gdf['P1_001N'].isna().sum()} blocks.")
gdf['P1_001N'] = gdf['P1_001N'].astype(int)
total_pop = int(gdf['P1_001N'].sum())


G = build_adjacency_graph(gdf)
return gdf, G, total_pop
