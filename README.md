# The Gerrymandering Problem

This project solves the complex problem of creating political districts from a set of census blocks. The goal is to generate districts that are fair by ignoring everything except Population, Compactness, and Contiguity - this removes all human bias.


---

### **Inputs**

* "2020.pl.zip" State File From: https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/
* "tabblock20.zip" State File From: https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/

---

### **Constraints**

### 1. Contiguity
- **Technical Summary**: A district is valid only if the subgraph induced by its member census blocks is connected. The program enforces this by creating a graph `G` where blocks are nodes and adjacency defines edges. For a set of blocks `d` in a district, the check `networkx.is_connected(G.subgraph(d))` must return `True`.
- **Plain English Summary**: Every part of a district must be physically connected. You should be able to "walk" from any point in a district to any other point without ever stepping into a different district. It ensures a district is one solid piece.

---
### 2. Population Parity
- **Technical Summary**: The population of each district ($P_d$) must be within a small tolerance of the ideal population ($P_{ideal}$). The program enforces the formula: $|P_d - P_{ideal}| \le P_{ideal} \times 0.005$. This means the population must be within ±0.5% of the target.
- **Plain English Summary**: Each district must have almost the exact same number of people. It's like cutting a cake for a group of people and making sure every single slice is the same size.

---
### 3. Shape Compactness
- **Technical Summary**: The geometric compactness of a district is measured using the Polsby-Popper score, calculated with the formula: $$PP(D) = \frac{4\pi \times Area(D)}{Perimeter(D)^2}$$ A score of 1.0 represents a perfect circle. The program requires that for every district `D`, $PP(D) \ge compactness\_threshold$ (e.g., 0.20).
- **Plain English Summary**: This rule prevents districts that are long, thin, and snaky. It encourages shapes that are more bundled and "compact," like a puddle rather than a winding river, making them fairer and easier for a community to identify with.

---
### 4. Deterministic Output
- **Technical Summary**: The program guarantees an identical output for a given input by eliminating all randomness. It creates a deterministic "sweep order" for processing census blocks by calculating a `hashlib.sha256` digest of the sorted block IDs, which seeds a deterministic sorting algorithm. All tie-breaking decisions are also handled by sorting, ensuring a consistent choice is made every time.
- **Plain English Summary**: The map is drawn by a robot following a precise, unchangeable set of instructions. Because there's no human bias or random chance involved, the robot will draw the exact same map every single time it's given the same state data.
---

### **The Algorithm**

### 1. Data Loading and Preparation
- **Technical Summary**: The program begins by unzipping the TIGER/Line shapefile and the PL 94-171 census data zips for the selected state. It then loads the `.shp`, geoheader, and population files using `geopandas` and `pandas`, merging them into a single GeoDataFrame where each census block's geometry is linked to its population count.
- **Plain English Summary**: First, the program gathers two key pieces of information: a detailed map of all the neighborhood blocks and a census list saying how many people live in each block. It then combines them into a single master map that knows the shape and population of every single block.

---
### 2. Graph Construction
- **Technical Summary**: Using the merged GeoDataFrame, the program constructs an adjacency graph with `networkx`. Each census block is a node in the graph, and an edge is created between any two nodes whose corresponding blocks touch geographically. This graph mathematically represents the state's spatial layout.
- **Plain English Summary**: The program creates a "neighbor list" for every block on the map. It goes through the entire state and notes down every single block that touches another one. This creates a network of connections, like a social network but for geography.

---
### 3. Initial District Assignment
- **Technical Summary**: The program iterates through every block in a deterministic "sweep" order. It uses a greedy algorithm to assign each block to the most suitable adjacent district, prioritizing districts with the lowest population while staying within the maximum population tolerance. This creates a complete, valid first draft of the district map.
- **Plain English Summary**: The program starts with a blank map and a set of empty "buckets," one for each district. Following a fixed path across the state (e.g., top-to-bottom), it picks up each block one-by-one and places it into the best neighboring bucket that isn't too full yet.

---
### 4. Optimization
- **Technical Summary**: The initial map is refined using a greedy local search algorithm. The program repeatedly evaluates moving a single block from its current district to a neighboring one. If a move improves the overall compactness score without violating any constraints (contiguity, population parity), the best such move is applied. This process repeats until no more improvements can be found.
- **Plain English Summary**: The program polishes the first-draft map. It looks for any blocks along the edges of districts that might be a better fit in the district next door. It carefully moves these blocks, one by one, to make the districts more compact and tidy, stopping when no more improvements can be made.

---
### 5. Finalization
- **Technical Summary**: Once the optimization is complete, the final district assignments are saved. A `.png` image of the new district map is generated using `matplotlib` for visualization, and a `.json` file containing the precise list of census blocks for each district, along with final scores and population counts, is written to the disk.
- **Plain English Summary**: After the map is polished, the program takes a "picture" of it and saves it as an image file. It also creates a detailed text file that lists exactly which blocks belong to each new district, creating a permanent record of the final, fair map.


---
---
---

# Stop Gerrymandering — Modular Redistricting Tool

## Build & Run

### 1. Clone the repo
git clone https://github.com/gitdeath/stopgerrymandering.git

cd stopgerrymandering

### 2. Set up Python environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

### 3. Install dependencies
pip install -e .

If GeoPandas stack fails on Windows:
pip install --only-binary=:all: shapely pyproj fiona rtree geopandas

### 4. Prepare input data
Place in project root:
- tl_2024_<FIPS>_tabblock20.zip (TIGER/Line block shapefile)
- <state>2020.pl.zip (PL 94-171 data, e.g. mo2020.pl.zip)

### 5. Run the tool
redistrict --state MO --config config/states.yaml --debug

- --state = two-letter state code (e.g. MO, CA, TX)
- --config = path to the YAML config
- --debug = optional flag for detailed logs

---

## Development Notes
- Config (config/states.yaml) defines CRS, tolerances, compactness thresholds, and state metadata.
- CLI lives in src/redistricting/cli.py.
- Modules are split for I/O, graph building, metrics, assignment, optimization, and visualization.
