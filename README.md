# The Gerrymandering Problem

This project solves the complex problem of creating political districts from a set of census blocks. The goal is to generate districts that are fair by ignoring everything except Population, Compactness, and Continguity - this removes all human bias.


---

### **Inputs**

* "2020.pl.zip" State File From: https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/
* "tabblock20.zip" State File From: https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/

---

### **Constraints**

To ensure the resulting districts are legally compliant and geometrically sound, the following constraints must be met:

* **Contiguity:** Each district must be a single, unbroken area.
* **Population Parity:** The population of each district must be within ±0.5% of the ideal population, calculated as `P_ideal = (Total Population) / D`.
* **Shape Compactness:** To prevent the creation of long, winding districts, each district's shape is measured using the **Polsby-Popper score**. This score is a ratio of a district's area to the square of its perimeter, with a perfect circle having a score of 1. Our solution requires each district to have a score of at least 0.20:
    `PolsbyPopper_d = (4 * pi * Area_d) / (Perimeter_d^2) >= 0.20`

---

### **Objective**

The primary objective is to minimize the **total population-weighted moment of inertia** (`sum J_d`) across all districts. This metric quantifies how compact the districts are in relation to their population distribution, prioritizing configurations where the population is close to the district's center.

For each district `d`:

1.  **Population Centroid (`mu_d`):** This is the population-weighted center of the district, calculated as:
    `mu_d = ( (sum p_i * x_i) / (sum p_i), (sum p_i * y_i) / (sum p_i) )`
2.  **Moment of Inertia (`J_d`):** This measures how far the population of a district is from its centroid. A lower value indicates a more compact district.
    `J_d = sum for i in district d [ p_i * distance((x_i,y_i), mu_d)^2 ]`

---

### **The Algorithm**

Our approach uses a multi-step, deterministic algorithm to find an optimal districting plan.

#### **1. Initial Assignment: Quadrant Sweep**

First, the algorithm establishes a deterministic sweep order for assigning census blocks.

* **Bounding Box:** The algorithm computes the geographical boundaries of the entire state to define a sweeping area.
    * `min_x = min(x_i)`, `max_x = max(x_i)`
    * `min_y = min(y_i)`, `max_y = max(y_i)`
* **Sweep Order:** A deterministic **tie-breaker hash** (see below) is used to select one of four sweep orders:
    * **0: Northeast Sweep** (descending y, ascending x)
    * **1: Southwest Sweep** (ascending y, ascending x)
    * **2: Southeast Sweep** (ascending y, descending x)
    * **3: Northwest Sweep** (descending y, descending x)
* The algorithm then assigns blocks to initial districts by iterating through them in the determined sweep order, merging adjacent blocks until the ideal population is met.

#### **2. Optimization: Hill-Climbing / Simulated Annealing**

After the initial district assignment, the algorithm refines the boundaries to improve the compactness score while maintaining all constraints.

* A **hill-climbing** or **simulated annealing** approach is used to iteratively move blocks between adjacent districts.
* At each step, a move (swapping a block) is evaluated. The move is accepted if it reduces the total moment of inertia (`sum J_d`) and does not violate any of the contiguity, population, or Polsby-Popper constraints.
* This process continues until no further improvements can be made.

#### **3. Tie-Breaker Rules**

To ensure a single, reproducible result even when multiple optimal solutions exist, a series of tie-breaker rules are applied. These rules are used to select the quadrant sweep order and to resolve any remaining ties in the final output.

1.  **Quadrant Sweep Selection:** A **SHA256 hash** of the concatenated, sorted block IDs is computed. The first byte of this hash, modulo 4, selects the sweep order (0-3).
2.  **Lexicographical Block ID:** If multiple solutions have the same compactness score, the solution is chosen based on the lexicographical ordering of the block IDs within each district.
3.  **Lexicographical Centroid Ordering:** If a tie still exists, a final tie-breaker is applied by comparing the lexicographical ordering of the district centroids (sorted by x, then y coordinates).

---

### **Output**

* `district_map`: A list of districts, where each district is a set of the block IDs it contains.
* `final_compactness_score`: The total moment of inertia (`sum J_d`) of the final districting plan.


