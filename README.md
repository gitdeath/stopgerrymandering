# The Gerrymandering Problem

This project solves the complex problem of creating political districts from a set of census blocks. The goal is to generate districts that are fair by ignoring everything except Population, Compactness, and Continguity - this removes all human bias.


---

### **Inputs**

* "2020.pl.zip" State File From: https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/
* "tabblock20.zip" State File From: https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/

---

### **Constraints**

**1. Contiguity**  
Every district must consist of a single, connected geographic area. No district may contain disconnected pieces.  

**Explanation:** A district has to be one unbroken shape on the map—no scattered islands or separate chunks.  

**2. Population Parity**  
Each district’s population must be within ±0.5% of the ideal size, where  
`P_ideal = (Total Population) / D`.  

**Explanation:** All districts must have nearly the same number of people. We calculate the perfect size by dividing the total population by the number of districts, and then make sure every district is within half a percent of that number.  

**3. Shape Compactness**  
District shapes are evaluated with the **Polsby-Popper score**, defined as:  
`PolsbyPopper_d = (4 * π * Area_d) / (Perimeter_d^2)`  
Each district must score at least 0.20. 

**Explanation:** To stop oddly stretched or “squiggly” shapes, we check how round each district is using a compactness score. A perfect circle is 1, and we require at least 0.20 to make sure districts are reasonably tidy and not distorted.  

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
* If the move **increases** inertia, it may still be accepted with a probability that decreases over time (the "cooling schedule").  

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


