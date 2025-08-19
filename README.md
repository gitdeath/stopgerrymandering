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

<br>

**1. Initial Assignment: Quadrant Sweep**

First, the algorithm establishes a deterministic sweep order for assigning census blocks.

* **Bounding Box**: The algorithm computes the geographical boundaries of the entire state to define a sweeping area:
    $$min\_x = \min(x_i), \quad max\_x = \max(x_i)$$
    $$min\_y = \min(y_i), \quad max\_y = \max(y_i)$$

* **Sweep Order**: A **deterministic tie-breaker hash** (see below) is used to select one of four sweep orders:
    * 0: Northeast Sweep (descending y, ascending x)
    * 1: Southwest Sweep (ascending y, ascending x)
    * 2: Southeast Sweep (ascending y, descending x)
    * 3: Northwest Sweep (descending y, descending x)
    The algorithm then assigns blocks to initial districts by iterating through them in the determined sweep order, merging adjacent blocks until the ideal population is met.

<br>

**2. Optimization: Simulated Annealing**

After the initial district assignment, the algorithm refines the boundaries to improve the compactness score while maintaining all constraints.

Our approach uses a **simulated annealing** method to iteratively move blocks between adjacent districts.

At each step, a move (swapping a block) is evaluated by calculating its effect on the total **moment of inertia** ($$\Sigma J_d$$).

* **Improving Moves**: The move is accepted if it reduces the total moment of inertia and does not violate any of the contiguity, population, or Polsby-Popper constraints.
* **Worsening Moves**: A move that increases the score (makes the map worse) may still be accepted with a certain probability. This probability decreases over time according to a "cooling schedule."  This process allows the algorithm to escape **local optima** and explore a wider range of possible solutions, increasing its chance of finding a globally optimal plan. The process continues for a fixed number of iterations.

<br>

**3. Tie-Breaker Rules**

To ensure a single, reproducible result, a set of deterministic tie-breaker rules are applied throughout the process.

* **Quadrant Sweep Selection**: A **SHA256 hash** of the concatenated, sorted block IDs is computed. The first byte of this hash, modulo 4, selects the sweep order (0-3).
* **Initial Assignment Tie-Breaker**: During the initial sweep, if a block can be legally assigned to multiple districts, the algorithm deterministically chooses the district with the **lowest current population**. In the case of a tie in population, it chooses the district with the **lowest index**.
* **Optimization Tie-Breaker**: The simulated annealing process is made fully reproducible by **seeding the random number generator** with the number of districts. This ensures that every run of the program for a given state and number of districts will produce the exact same sequence of "random" numbers and, therefore, the same outcome.
* **Final Output Ordering**: The final output districts are sorted based on their contents to guarantee the final JSON file is consistently formatted.

***

### Layperson Summary

This redistricting program creates a fair and compact map using a consistent, repeatable process.

First, the program creates an initial draft of the districts. It does this by drawing an invisible box around the state and, using a special digital code based on the census data, chooses a specific starting corner to "sweep" from. As it sweeps, it assigns census blocks to districts one by one, making sure each district meets its target population.

Next, the program refines this initial draft using a technique called **simulated annealing**. This process is like a more sophisticated version of "trial and error." It makes small, random changes to the district boundaries. It always keeps a change that makes the districts better (more compact), but it will also sometimes, early in the process, accept a change that makes the districts a little worse. This counterintuitive step is crucial because it helps the program avoid getting stuck in a mediocre solution and find a truly great one.

Finally, to make sure the result is always the same, the program uses a set of tie-breaker rules. It uses the digital code to select the initial sweep direction, ensuring it always starts the same way. The "trial and error" phase is also made repeatable by using the number of districts as a "seed," which ensures the same sequence of random choices is made every time. This guarantees that you will get the exact same, high-quality map no matter how many times you run the program on the same data.

### **Output**

* `district_map`: A list of districts, where each district is a set of the block IDs it contains.
* `final_compactness_score`: The total moment of inertia (`sum J_d`) of the final districting plan.


