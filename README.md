INPUT:
    census_blocks = set of all census blocks in state
        each block has: population p_i, coordinates (x_i, y_i), adjacency list
    D = number of districts (fixed by law)

CONSTRAINTS:
    - Each district must be contiguous
    - Each district population within ±0.5% of ideal (total_pop / D)

OBJECTIVE:
    Minimize total population-weighted moment of inertia:
        For each district d:
            μ_d = ( Σ (p_i * x_i), Σ (p_i * y_i) ) / Σ p_i   # population centroid
            J_d = Σ p_i * distance((x_i,y_i), μ_d)^2
        Global objective: Minimize Σ_d J_d

SECONDARY CONSTRAINT (shape guardrail):
    For each district d:
        PolsbyPopper_d = 4π * Area_d / (Perimeter_d^2)
        Require PolsbyPopper_d ≥ 0.20
        # prevents long snakes / tendrils even if inertia is low

DETERMINISTIC QUADRANT SWEEP:
    - Compute bounding box of all census blocks:
        min_x = min(x_i), max_x = max(x_i)
        min_y = min(y_i), max_y = max(y_i)
    - Assign sweep order based on tie-breaker hash:
        0 → NE sweep  = descending y, ascending x
        1 → SW sweep  = ascending y, ascending x
        2 → SE sweep  = ascending y, descending x
        3 → NW sweep  = descending y, descending x
    - Sweep order determines which blocks are assigned first when forming districts

TIE-BREAK RULES:
    1. Compute SHA256 hash of concatenated sorted block IDs
    2. First byte of hash mod 4 → select quadrant sweep order (as above)
    3. If multiple maps tie after orientation selection:
         lexicographic ordering of block IDs
    4. If still tied:
         lexicographic ordering of centroids (sorted by x,y)

ALGORITHM:
    Step 1: Assign blocks to an initial grid scan based on chosen quadrant sweep
    Step 2: Merge adjacent blocks into districts until reaching ideal population
    Step 3: Optimize assignment using hill-climb or simulated annealing to minimize Σ J_d
            subject to contiguity + population + PolsbyPopper ≥ 0.20
    Step 4: Apply tie-breaker rules if multiple optimal solutions exist
    Step 5: Output district boundaries

OUTPUT:
    district_map = list of districts
        each district = set of block IDs
    final_compactness_score = Σ J_d
