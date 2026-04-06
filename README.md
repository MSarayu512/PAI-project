# PAI Urban Agent: City Decision Dashboard

**Team Members:** Divya Battu (230968042), M Sarayu (230968052)

A Rational AI Agent simulation environment for city decision-making tasks, built as part of the Principles of Artificial Intelligence Lab. This project leverages classical AI search algorithms, constraint satisfaction, adversarial reasoning, and reinforcement learning on actual real-world transit layouts to demonstrate optimal operations in dynamically changing urban environments.

## Overview

Urban environments represent highly complex state spaces involving varied population densities, restricted infrastructure, and conflicting resource demands. Traditional urban planning relies heavily on manual heuristics.

Our project builds an **Explainable Rational AI Agent** capable of operating within these constraints. Using OpenStreetMap data (`osmnx`), the agent initializes an authentic representation of a city (modelled around MG Road, Bangalore). The system provides structured and highly transparent decision-making outputs for efficient routing, strategic facility placement, adversarial bottleneck prediction, and ongoing dynamically learned policies, all visualized via a rich frontend interface.

## Detailed Synopsis

Modern urban contexts create constant challenges in traffic routing and maintaining population-to-resource equity. Automated decision tools usually lack explainability, leaving authorities guessing about systemic behaviors.

To address these pain points, the PAI Urban Agent is tasked with mimicking distinct city-planning challenges using varied AI paradigms:
1. **Topological Graph Search:** Traditional search algorithms deployed to resolve point-to-point transit optimization.
2. **Facility Allocation:** Modeling public utility distributions (like hospitals or charging stations) as Constraint Satisfaction Problems (CSP).
3. **Traffic Anomalies:** Modeling unpredictable road conditions mapping as adversarial bottlenecks using Minimax algorithms.
4. **Adaptive Navigation:** Q-Learning models that can dynamically respond to long-term traffic fluctuations and policy shifts without recalculating trees from scratch.

This hybrid approach integrates theoretical AI systems into a unified framework serving real architectural layouts.

---

## Technical Implementation Details

The system is organized into a Python/Flask-driven backend (`app.py`), and a vanilla JavaScript interactive frontend (`frontend_index.html`).

### 1. Connectivity & Graph Infrastructure
- **Network Construction:** The backend uses `osmnx` to query OpenStreetMap and generate a driving network within 1.5km of MG Road, Bangalore. Unconnected streets are discarded (retaining only the largest strongly connected component) to guarantee valid search paths.
- **Node Meta:** The graph maps each node to uniformly scaled geographical bounds (so the UI correctly renders aspect ratios) and structurally designates arbitrary `zone` characteristics (residential, commercial, industrial) alongside population weights.
- **REST Connectivity:** Python leverages standard Flask routing combined with `flask_cors`. Endpoints (`/api/search`, `/api/csp`, etc.) respond uniformly via JSON containing not just the optimal path data, but a structured list explaining the logic, node expansions, and internal timings allowing the frontend to narrate the decision.

### 2. Implementation of AI Paradigms

The core strength of the project is the execution and formal logic applied within `app.py`.

#### Standard Search Routing (`/api/search`)
Used to traverse the geographical node space:
- **Breadth-First Search (BFS):** Implemented using Python's `collections.deque`. Guarantees discovering the path with the fewest minimum spatial hops, but remains strictly oblivious to weighted real-world traffic costs.
- **Uniform Cost Search (UCS / Dijkstra's):** Employs standard priority queue expansion (`heapq`). By rigidly accumulating traversing cost ($g$), UCS always converges upon the realistically cheapest transit route.
- **A* Algorithm:** Bolsters the UCS priority logic with an admissible Euclidean distance heuristic ($h$), computed via `math.hypot()` on precise geographical coordinates. This rigorously prunes counter-directional branches, vastly suppressing expanded nodes while securely returning the optimal travel path.

#### Constraint Satisfaction Problem (`/api/csp`)
Addresses the logistical placement of multiple key urban facilities (like a hospital or power station):
- **Core Strategy:** Backtracking Depth-First Search with hard constraint pruning.
- **Domain Restraints:** Initialized to only permit facility generation on nodes preassigned as `medical`, `commercial`, or `residential` zones.
- **Hard Constraints:** Facilities cannot be clustered together. Defined by a dynamically tested `min_separation` Euclidean distance.
- **Optimisation:** For every valid depth loop that matches the required number of placements ($k$), the algorithm checks population spheres—scoring the layout based on total aggregated citizens covered. The permutation with the highest geographical coverage is finalized.

#### Adversarial Reasoning (`/api/adversarial`)
Evaluates worst-case scenarios via competitive adversarial games. Traffic or malicious blockage is treated as a competing opponent determining bottlenecks.
- **System Dynamics:** A limited recursive Minimax tree formulation restricted by depth and subgraph boundaries.
- **Actor (Maximizing Agent):** Systematically attempts to identify the smallest total path weight toward the goal to minimize delays. 
- **Adversary (Minimizing Opponent):** Attempts to maximize structural delay by arbitrarily inflating critical edge traffic weights by $2.2\times$, purposefully targeting the agent's expected path.
- **Performance Output:** Employs explicit $\alpha-\beta$ pruning bounds to cleanly sever computational branches where an adversarial traffic jam has already mathematically forced a route abandonment. 

#### Reinforcement Learning - Q-Learning (`/api/rl`)
Simulates long-term behavioral adaptation without manual routing.
- **Structure:** Tabular formulation modeling routes to a single persistent destination `Goal`.
- **Dynamic Environment Variables:** The world continuously modifies edge path values randomly ($\pm 15\%$) every 60 continuous episodes. This simulates recurring traffic fluctuations or road deterioration in real life.
- **Parametrics:** Trained using learning rate $\alpha = 0.3$, discount factor $\gamma = 0.9$, and an exploration constant $\epsilon = 0.25$.
- **Reward Engine:** Yields negative step-penalties matching local edge weights (preventing excessive wandering), with an ultimate $+50$ utility boost for successfully locating the active goal coordinate. Over sufficient epochs, the Q-table uniformly converges on flexible, congestion-avoidant routes.
