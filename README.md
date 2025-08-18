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
    * **Explanation:** Every new district has to be one continuous shape on the map, without any disconnected pieces or "islands."
    
* **Population Parity:** The population of each district must be within ±0.5% of the ideal population, calculated as `P_ideal = (Total Population) / D`.
    * **Explanation:** The population in each new district must be almost exactly the same. We figure out what the ideal population is by dividing the total population by the number of districts, and then we make sure no district is more than a tiny bit (0.5%) bigger or smaller than that ideal number. This ensures a fair distribution of voters.
      
* **Shape Compactness:** To prevent the creation of long, winding districts, each district's shape is measured using the **Polsby-Popper score**. This score is a ratio of a district's area to the square of its perimeter, with a perfect circle having a score of 1. Our solution requires each district to have a score of at least 0.20:
    `PolsbyPopper_d = (4 * pi * Area_d) / (Perimeter_d^2) >= 0.20`

    * **Explanation:** We use a special formula called the Polsby-Popper score to check how "round" or compact a district's shape is. A score of 1 is a perfect circle, which is the most compact shape. A very low score means the district is long and squiggly, which often indicates gerrymandering. We require each district to have a score of at least 0.20, making sure its shape is reasonably tidy and not too stretched out. 
---

### **Objective**

The primary objective is to minimize the **total population-weighted moment of inertia** (`sum J_d`) across all districts. This metric quantifies how compact the districts are in relation to their population distribution, prioritizing configurations where the population is close to the district's center.

For each district `d`:

1.  **Population Centroid (`mu_d`):** This is the population-weighted center of the district, calculated as:
    `mu_d = ( (sum p_i * x_i) / (sum p_i), (sum p_i * y_i) / (sum p_i) )`

    **Explanation:** Imagine you have a map of a district on a seesaw, and each person in that district is a small weight on the seesaw at their home's location. The population centroid is the exact point where you would need to place the seesaw's pivot so that it balances perfectly.
    
3.  **Moment of Inertia (`J_d`):** This measures how far the population of a district is from its centroid. A lower value indicates a more compact district.
    `J_d = sum for i in district d [ p_i * distance((x_i,y_i), mu_d)^2 ]`

     **Explanation:** This is like a "spread-out" score. The moment of inertia tells us how far away, on average, all the people in a district are from its balancing point (the centroid). We want this score to be as low as possible for all districts combined, because a low score means the population is close together. This creates districts that are more compact and sensible for the people living in them.
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

**Explanation:** The process begins by using a predetermined sweeping direction to create the first version of the districts. It draws a box around the entire state and then, based on a unique hash code, decides to sweep from one of the four corners (northeast, southwest, southeast, or northwest). As it sweeps, it adds adjacent census blocks to a district until that district has the correct number of people.

#### **2. Optimization: Hill-Climbing / Simulated Annealing**

After the initial district assignment, the algorithm refines the boundaries to improve the compactness score while maintaining all constraints.

* A **hill-climbing** or **simulated annealing** approach is used to iteratively move blocks between adjacent districts.
* At each step, a move (swapping a block) is evaluated. The move is accepted if it reduces the total moment of inertia (`sum J_d`) and does not violate any of the contiguity, population, or Polsby-Popper constraints.
* This process continues until no further improvements can be made.

**Explanation:** Once the initial districts are formed, the algorithm refines them by using a process that’s like making small improvements. It repeatedly considers moving a single block from one district to a neighboring one. If this move makes the districts more compact and doesn't break any of the population or shape rules, it accepts the change. This continues until no such moves can be made to improve the districts further.

#### **3. Tie-Breaker Rules**

To ensure a single, reproducible result even when multiple optimal solutions exist, a series of tie-breaker rules are applied. These rules are used to select the quadrant sweep order and to resolve any remaining ties in the final output.

1.  **Quadrant Sweep Selection:** A **SHA256 hash** of the concatenated, sorted block IDs is computed. The first byte of this hash, modulo 4, selects the sweep order (0-3).
2.  **Lexicographical Block ID:** If multiple solutions have the same compactness score, the solution is chosen based on the lexicographical ordering of the block IDs within each district.
3.  **Lexicographical Centroid Ordering:** If a tie still exists, a final tie-breaker is applied by comparing the lexicographical ordering of the district centroids (sorted by x, then y coordinates).

**Explanation:** These are rules designed to ensure a consistent outcome every time the process is run. If the algorithm finds multiple equally good solutions, a set of rules are applied to decide which one to choose. This includes using a unique digital code (a SHA256 hash) to select the starting sweep direction and then sorting the districts by their block IDs or geographic centers to break any remaining ties.

---

### **Output**

* `district_map`: A list of districts, where each district is a set of the block IDs it contains.
* `final_compactness_score`: The total moment of inertia (`sum J_d`) of the final districting plan.


