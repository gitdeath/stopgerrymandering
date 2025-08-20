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

### **Objective**

The goal is to minimize the **total population-weighted moment of inertia** (`Σ J_d`) across all districts. This ensures districts are compact with populations centered close to their district’s centroid.  

For each district `d`:  

**1. Population Centroid (`μ_d`)**  
The population-weighted center of the district is calculated as:  
`μ_d = ( (Σ p_i * x_i) / (Σ p_i), (Σ p_i * y_i) / (Σ p_i) )`  

**Explanation:** Imagine balancing a district on a seesaw where each person is a small weight at their home location. The centroid is the exact point where it would balance perfectly.  

**2. Moment of Inertia (`J_d`)**  
Measures how spread out the population is around the centroid:  
`J_d = Σ [ p_i * distance((x_i, y_i), μ_d)^2 ]`  

**Explanation:** This is like a "spread score." A low value means most people live close to the center of the district, creating a compact, sensible shape. High values mean the population is scattered, making the district less compact.  

The algorithm seeks to minimize the sum of these inertia values across all districts: 
`Σ J_d`  

**Explanation:** By keeping the spread small in every district, the final map is fairer, with each district drawn around its natural center of population.  

---

### **The Algorithm**

The districting process is performed in three main steps: **Initial Assignment**, **Optimization**, and **Tie-Breaker Rules**.  

---

**1. Initial Assignment: Quadrant Sweep**  
The algorithm begins by assigning census blocks to districts in a deterministic sweep order.  

* **Bounding Box:**  
  Define the state’s limits:  
  `min_x = min(x_i),  max_x = max(x_i)`  
  `min_y = min(y_i),  max_y = max(y_i)`  

* **Sweep Order:**  
  A SHA256 hash of the sorted block IDs is used. The first byte of the hash, modulo 4, selects one of four sweep directions:  
  - 0: Northeast (descending y, ascending x)  
  - 1: Southwest (ascending y, ascending x)  
  - 2: Southeast (ascending y, descending x)  
  - 3: Northwest (descending y, descending x)  

  Blocks are then assigned one by one until the ideal population is reached for each district.  

**Explanation:** The program picks a corner of the state to start from (chosen consistently using a digital hash of the data). It then sweeps across, filling districts with blocks until each one reaches its target population.  

---

**2. Optimization: Simulated Annealing**  
Once the draft districts are created, the boundaries are refined using **simulated annealing**.  

At each step, a block may be moved between neighboring districts:  
* If the move **reduces** total inertia (`Σ J_d`) and satisfies all constraints (contiguity, population, compactness), it is accepted.  
* If the move **increases** inertia, it may still be accepted with a probability that decreases over time (the "cooling schedule" `temp = 1000, alpha = 0.95, max_iter = 10000`).  

This allows exploration of many possible solutions and avoids getting stuck in suboptimal arrangements.  

**Explanation:** The program improves the draft map by trial and error. It usually keeps good changes that make districts more compact, but sometimes—especially early—it also keeps worse changes to avoid getting stuck with a mediocre map. Over time, it becomes pickier, keeping only better solutions.  

---

**3. Tie-Breaker Rules**  
To guarantee reproducibility, deterministic tie-breakers are used:  

* **Sweep Direction:** Chosen from the SHA256 hash of block IDs.  
* **Block Assignment:** If multiple districts are possible, choose the one with the lowest current population; ties go to the lowest index.  
* **Randomization:** Simulated annealing uses a seeded random number generator (seed = number of districts).  
* **Final Output:** Districts are sorted consistently for stable JSON output.  

**Explanation:** These rules ensure the program always produces the same map when given the same data, eliminating randomness and guaranteeing fairness and repeatability.  


### **Output**

* `district_map`: A list of districts, where each district is a set of the block IDs it contains.
* `final_compactness_score`: The total moment of inertia (`sum J_d`) of the final districting plan.


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

### 6. Outputs
- districts_MO.png — district map visualization
- districts_MO.json — district assignments and metrics

---

## Development Notes
- Config (config/states.yaml) defines CRS, tolerances, compactness thresholds, and state metadata.
- CLI lives in src/redistricting/cli.py.
- Modules are split for I/O, graph building, metrics, assignment, optimization, and visualization.
