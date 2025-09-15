MO-DMLHM: Multi-Objective Dynamic Hypergraph Modeling for Community Detection
Project Overview
MO-DMLHM (Multi-Objective Dynamic Hypergraph Modeling) is an algorithmic framework for cross-layer community detection in dynamic multi-layer organizational networks. By integratingdynamic hypergraph modeling,multi-objective optimization, andhybrid encoding evolutionary algorithms, it addresses limitations of traditional methods in handling temporal dynamics, cross-layer interactions, and multi-objective conflicts. Core innovations include:
Adaptive Dynamic Hypergraph Modeling: Captures spatiotemporal dynamics via dual-scale weight decay and time window partitioning.
Four-Dimensional Multi-Objective Optimization: Balances hypergraph modularity, cross-layer consistency, dynamic stability, and resource efficiency.
Hybrid Encoding Evolutionary Algorithm: Jointly optimizes hyperedge activation (topology) and node community assignment (membership).
Core Features
1. Dynamic Hypergraph Construction (DynamicHypergraphClass)
Hyperedge Weight Decay: Applies distinct decay rates for transient interactions (λshort=0.4) and strategic partnerships (λlong=0.15) to model relationship timeliness.
New Event Detection: Dynamically generates new hyperedges to simulate emerging interactions (e.g., ad-hoc meetings, project collaborations).
Adjacency Matrix & Coupling Tensor Updates: Quantifies intra-layer node correlations and cross-layer community alignment, enabling multi-layer network analysis.
2. Multi-Objective Function Evaluation
Hypergraph Modularity (Qh): Measures community cohesion by comparing intra-community vs. inter-community hyperedge density.
Cross-Layer Consistency (Jc): Evaluates alignment of community structures across layers using Jaccard similarity and Normalized Mutual Information (NMI).
Dynamic Stability (Sd): Balances temporal continuity and adaptability via NMI and reorganization penalties to suppress abrupt community overhauls.
Resource Efficiency (Re): Quantifies operational costs of cross-layer collaboration, considering node role hierarchy and layer span.
3. NSGA-III Community Optimization (NSGAIIICommunityDetectorClass)
Hybrid Encoding: Combines binary structural genes (hyperedge activation) and integer community genes (node assignment) for joint topology-membership optimization.
Non-Dominated Sorting: Partitions the population into Pareto fronts to identify optimal solutions balancing conflicting objectives.
Environmental Selection: Preserves solution diversity using reference point-based distance metrics, enhancing optimization robustness.
Installation
Dependencies
numpy(numerical computing)
scipy(scientific computing, e.g., exponential decay)
scikit-learn(evaluation metrics like NMI)
deap(evolutionary algorithm framework for crossover/mutation)
Installation Command
pip install numpy scipy scikit-learn deap
Usage Example
1. Data Preparation
import numpy as np
import random
from dynamic_hypergraph import DynamicHypergraph
from nsga3_detector import NSGAIIICommunityDetector
# Generate synthetic data
num_nodes = 30
nodes = [f"node_{i}" for i in range(num_nodes)] # Node list: node_0 to node_29
layers = ["R&D", "Marketing", "Management"] # Functional layers
initial_hyperedges = {}
for i in range(20):
he_id = f"he_initial_{i}" # Hyperedge ID
selected_nodes = random.sample(nodes, k=4) # 4 nodes per hyperedge
layer = layers[i % 3] # Distribute hyperedges across layers
initial_hyperedges[he_id] = (
{'nodes': selected_nodes, 'type': 'strategic'}, # Strategic partnership
layer,
0, # Creation time (initial time window)
1.0 # Initial weight
)
time_windows = [0, 1, 2, 3] # Time window sequence
2. Dynamic Hypergraph Construction
# Initialize hypergraph builder
hypergraph_builder = DynamicHypergraph(
nodes=nodes,
layers=layers,
initial_hyperedges=initial_hyperedges,
time_windows=time_windows
)
# Construct hypergraph at time window=1
hypergraph_data = hypergraph_builder.construct_hypergraph(current_time=1)
print(f"Hypergraph Constructed at Time=1 | Hyperedges: {len(hypergraph_data[0])} | Layers: {len(hypergraph_data[1])}")
3. NSGA-III Community Optimization
# Initialize NSGA-III detector
nsga3_detector = NSGAIIICommunityDetector(
hypergraph=hypergraph_data,
pop_size=50, # Population size
generations=10, # Evolutionary generations
crossover_prob=0.8, # Crossover probability
mutation_prob=0.2 # Mutation probability
)
# Optimize community structure (max 5 communities)
optimal_individual, optimal_community = nsga3_detector.optimize(num_nodes=num_nodes, num_communities=5)
4. Output Results
# Print optimal community structure
print("\nOptimal Community Structure:")
for comm_id, nodes in optimal_community.items():
print(f"Community {comm_id}: {nodes}")
Parameter Specification
1. Dynamic Hypergraph Parameters (DynamicHypergraph)
Parameter	Description	Default	Tuning
lambda_short	Short-term interaction decay	0.4	Increase for noisy networks (e.g., 0.5)
lambda_long	Long-term partnership decay	0.15	Decrease for stable networks (e.g., 0.1)
prune_threshold	Hyperedge pruning threshold	0.1	Increase to retain more hyperedges (e.g., 0.2)
event_rate	New event generation rate	0.1	Increase for dynamic networks (e.g., 0.2)
2. NSGA-III Optimization Parameters (NSGAIIICommunityDetector)
Parameter	Description	Default	Tuning
pop_size	Population size	50	Increase for complex networks (e.g., 100)
generations	Evolutionary generations	10	Increase for slow convergence (e.g., 20)
crossover_prob	Crossover probability	0.8	Decrease to preserve diversity (e.g., 0.7)
mutation_prob	Mutation probability	0.2	Increase for local search (e.g., 0.3)

Notes

Data Input: Thedetect_new_eventsfunction simulates event generation. For real-world applications, replace it with actual network data (e.g., email interactions, project collaborations).
Hyperparameter Tuning: Decay rates (lambda_short/lambda_long) and evolutionary parameters should be adjusted based on dataset characteristics. Use grid search for optimal performance.
Computational Complexity: For large-scale networks (>1000 nodes), reducepop_sizeor enable parallel computing to accelerate optimization.
Community Count: Thenum_communitiesparameter should align with domain knowledge. Underspecification may cause underfitting, while overspecification increases computational load.

References

Li, Y. et al. (2025). MO-DMLHM: Multi-Objective Dynamic Hypergraph Modeling for Cross-Layer Community Detection in Organizational Networks.Information Sciences.

Deb, K. & Jain, H. (2014). An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach.IEEE Transactions on Evolutionary Computation.
