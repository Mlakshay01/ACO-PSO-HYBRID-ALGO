import networkx as nx
import numpy as np
import random
from sklearn.model_selection import train_test_split, KFold  
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

# Load the Facebook dataset
def load_facebook_data(file_path="g_plusAnonymized.csv"):
    G = nx.read_edgelist(file_path, nodetype=int)
    return G

# Split the graph into training and testing sets
def split_graph(G, test_size=0.2, random_state=42):
    edges = list(G.edges())
    nodes = list(G.nodes())
    
    # Split edges into train and test sets
    train_edges, test_edges = train_test_split(edges, test_size=test_size, random_state=random_state)
    
    # Create training graph
    G_train = nx.Graph()
    G_train.add_nodes_from(nodes)
    G_train.add_edges_from(train_edges)
    
    # Generate non-edges for testing (negative examples)
    non_edges = list(nx.non_edges(G))
    random.seed(random_state)
    non_edges_sample = random.sample(non_edges, len(test_edges))
    
    return G_train, test_edges, non_edges_sample

# Calculate topological features
def extract_features(G, node_pairs):
    features = []
    
    for u, v in node_pairs:
        # Common Neighbors
        common_neighbors = len(list(nx.common_neighbors(G, u, v)))
        
        # Jaccard Coefficient
        if len(set(G.neighbors(u))) + len(set(G.neighbors(v))) - common_neighbors > 0:
            jaccard = common_neighbors / (len(set(G.neighbors(u))) + len(set(G.neighbors(v))) - common_neighbors)
        else:
            jaccard = 0
        
        # Adamic-Adar Index
        aa_index = 0
        for neighbor in nx.common_neighbors(G, u, v):
            aa_index += 1 / np.log(G.degree(neighbor))
        
        # Preferential Attachment
        pref_attachment = G.degree(u) * G.degree(v)
        
        # Resource Allocation
        ra_index = 0
        for neighbor in nx.common_neighbors(G, u, v):
            ra_index += 1 / G.degree(neighbor)
            
        features.append([common_neighbors, jaccard, aa_index, pref_attachment, ra_index])
    
    return np.array(features)
class ACO:
    def __init__(self, G, num_ants=50, num_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100):
        self.G = G
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Heuristic importance
        self.evaporation_rate = evaporation_rate
        self.Q = Q          # Pheromone deposit factor
        
        # Initialize pheromone on all possible edges
        self.nodes = list(G.nodes())
        self.pheromone = {}
        for i in range(len(self.nodes)):
            for j in range(i+1, len(self.nodes)):
                u, v = self.nodes[i], self.nodes[j]
                self.pheromone[(u, v)] = 1.0
        
        # Communities detected
        self.communities = {}
    
    def run(self):
        best_solutions = []
        
        for iteration in range(self.num_iterations):
            all_paths = []
            
            # Each ant constructs a solution
            for ant in range(self.num_ants):
                path = self._construct_solution()
                all_paths.append(path)
            
            # Update pheromones
            self._update_pheromones(all_paths)
            
            # Find best solution in this iteration
            best_path = max(all_paths, key=len)
            best_solutions.append(best_path)
        
        # Extract communities and patterns from solutions
        self._extract_communities(best_solutions)
        
        return self.communities, self.pheromone
    
    def _construct_solution(self):
        # Start from a random node
        current_node = int(random.choice(self.nodes))  # Convert to int
        path = [current_node]
        visited = {current_node}
        
        # Construct path
        for _ in range(random.randint(2, 10)):  # Random path length
            candidates = []
            probabilities = []
            
            # Calculate probabilities for next nodes
            for neighbor in self.G.neighbors(current_node):
                if neighbor not in visited:
                    edge = tuple(sorted([current_node, neighbor]))
                    if edge[0] != current_node:
                        edge = (edge[1], edge[0])
                    
                    # Calculate attractiveness based on pheromone and degree
                    pheromone = self.pheromone.get((min(current_node, neighbor), max(current_node, neighbor)), 1.0)
                    heuristic = self.G.degree(neighbor)
                    
                    attractiveness = (pheromone ** self.alpha) * (heuristic ** self.beta)
                    
                    candidates.append(neighbor)
                    probabilities.append(attractiveness)
            
            # Select next node based on probabilities
            if candidates:
                total = sum(probabilities)
                if total > 0:
                    probabilities = [p/total for p in probabilities]
                    next_node = np.random.choice(candidates, p=probabilities)
                else:
                    next_node = random.choice(candidates)
                
                visited.add(next_node)
                path.append(next_node)
                current_node = next_node
            else:
                break
        
        return path
    
    def _update_pheromones(self, all_paths):
        # Evaporation
        for edge in list(self.pheromone.keys()):
            self.pheromone[edge] *= (1 - self.evaporation_rate)
        
        # Deposit new pheromones
        for path in all_paths:
            for i in range(len(path) - 1):
                u, v = int(path[i]), int(path[i+1])  # Convert to Python int
                edge = (min(u, v), max(u, v))
                
                # Check if edge exists in pheromone dictionary, if not add it
                if edge not in self.pheromone:
                    self.pheromone[edge] = 1.0
                
                # Deposit pheromone proportional to path quality
                self.pheromone[edge] += self.Q / (1 + len(path))
    
    def _extract_communities(self, solutions):
        # Simple community detection based on pheromone levels
        edges_with_pheromones = [(edge, pheromone) for edge, pheromone in self.pheromone.items()]
        sorted_edges = sorted(edges_with_pheromones, key=lambda x: x[1], reverse=True)
        
        # Take top edges as community indicators
        top_edges = sorted_edges[:int(len(sorted_edges) * 0.2)]  # Top 20%
        
        G_community = nx.Graph()
        for (u, v), _ in top_edges:
            G_community.add_edge(u, v)
        
        # Extract connected components as communities
        self.communities = list(nx.connected_components(G_community))
        
        return self.communities
class PSO:
    def __init__(self, num_particles=30, num_dimensions=5, max_iter=100, w=0.7, c1=1.5, c2=1.5):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions  # Number of features to optimize weights for
        self.max_iter = max_iter
        self.w = w          # Inertia weight
        self.c1 = c1        # Cognitive coefficient
        self.c2 = c2        # Social coefficient
        
        # Initialize particles
        self.particles = np.random.rand(num_particles, num_dimensions)
        self.velocities = np.random.rand(num_particles, num_dimensions) * 0.1
        
        # Initialize best positions
        self.pbest = self.particles.copy()
        self.pbest_fitness = np.zeros(num_particles)
        
        # Initialize global best
        self.gbest = np.zeros(num_dimensions)
        self.gbest_fitness = -np.inf
    
    def fitness_function(self, weights, X_train, y_train, X_val, y_val):
        # Calculate weighted sum of features
        train_scores = X_train.dot(weights)
        val_scores = X_val.dot(weights)
        
        # Normalize scores to [0, 1]
        train_scores = 1 / (1 + np.exp(-train_scores))
        val_scores = 1 / (1 + np.exp(-val_scores))
        
        # Calculate AUC on validation set
        auc_score = roc_auc_score(y_val, val_scores)
        
        return auc_score
    
    def optimize(self, X_train, y_train, X_val, y_val):
        # Initialize pbest_fitness values
        for i in range(self.num_particles):
            self.pbest_fitness[i] = self.fitness_function(
                self.particles[i], X_train, y_train, X_val, y_val)
        
        # Find initial gbest
        best_idx = np.argmax(self.pbest_fitness)
        self.gbest = self.pbest[best_idx].copy()
        self.gbest_fitness = self.pbest_fitness[best_idx]
        
        # Main PSO loop
        for iteration in range(self.max_iter):
            # Update velocities and positions
            for i in range(self.num_particles):
                # Random coefficients
                r1 = np.random.rand(self.num_dimensions)
                r2 = np.random.rand(self.num_dimensions)
                
                # Update velocity
                cognitive = self.c1 * r1 * (self.pbest[i] - self.particles[i])
                social = self.c2 * r2 * (self.gbest - self.particles[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                
                # Update position
                self.particles[i] += self.velocities[i]
                
                # Clip weights to [0, 1] for stability
                self.particles[i] = np.clip(self.particles[i], 0, 1)
                
                # Evaluate new position
                fitness = self.fitness_function(
                    self.particles[i], X_train, y_train, X_val, y_val)
                
                # Update personal best
                if fitness > self.pbest_fitness[i]:
                    self.pbest[i] = self.particles[i].copy()
                    self.pbest_fitness[i] = fitness
                    
                    # Update global best
                    if fitness > self.gbest_fitness:
                        self.gbest = self.particles[i].copy()
                        self.gbest_fitness = fitness
            
            # Print progress every 10 iterations
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Best fitness: {self.gbest_fitness:.4f}")
        
        return self.gbest, self.gbest_fitness

class ACO_PSO_Hybrid:
    def __init__(self, G, aco_params=None, pso_params=None):
        self.G = G
        
        # Default parameters if none provided
        if aco_params is None:
            aco_params = {
                'num_ants': 50,
                'num_iterations': 50,
                'alpha': 1.0,
                'beta': 2.0,
                'evaporation_rate': 0.5,
                'Q': 100
            }
        
        if pso_params is None:
            pso_params = {
                'num_particles': 30,
                'num_dimensions': 5,  # 5 features
                'max_iter': 100,
                'w': 0.7,
                'c1': 1.5,
                'c2': 1.5
            }
            
        self.aco_params = aco_params
        self.pso_params = pso_params
        
        self.aco = ACO(G, **aco_params)
        self.pso = None  # Will initialize after ACO
        
        self.optimal_weights = None
        self.communities = None
        self.pheromone_levels = None
    def _enhance_with_community_features(self, node_pairs):
        # Add community-based features
        features = np.zeros((len(node_pairs), 2))
        
        for i, (u, v) in enumerate(node_pairs):
            # Feature 1: Are nodes in same community?
            same_community = 0
            for community in self.communities:
                if u in community and v in community:
                    same_community = 1
                    break
            
            # Feature 2: Pheromone level between nodes
            pheromone = self.pheromone_levels.get((min(u, v), max(u, v)), 0)
            
            features[i] = [same_community, pheromone]
        
        return features
    def run(self, validation_size=0.3, random_state=42):
        print("Phase 1: Running ACO to detect communities and structure...")
        self.communities, self.pheromone_levels = self.aco.run()
        
        print(f"Detected {len(self.communities)} communities")
        
        print("Phase 2: Preparing features for PSO optimization...")
        # Split data into training and validation
        G_train = self.G.copy()
        
        # Get all edges
        edges = list(G_train.edges())
        
        # Split into train/validation
        train_edges, val_edges = train_test_split(edges, test_size=validation_size, random_state=random_state)
        
        # Remove validation edges from training graph
        for u, v in val_edges:
            G_train.remove_edge(u, v)
        
        # Sample non-edges for validation
        non_edges = list(nx.non_edges(G_train))
        val_non_edges = random.sample(non_edges, len(val_edges))
        
        # Extract features
        print("Extracting topological features...")
        X_train = extract_features(G_train, train_edges)
        y_train = np.ones(len(train_edges))
        
        # Add negative examples to training
        train_non_edges = random.sample(list(nx.non_edges(G_train)), len(train_edges))
        X_train_neg = extract_features(G_train, train_non_edges)
        y_train_neg = np.zeros(len(train_non_edges))
        
        X_train = np.vstack((X_train, X_train_neg))
        y_train = np.hstack((y_train, y_train_neg))
        
        # Validation set
        X_val = extract_features(G_train, val_edges)
        y_val = np.ones(len(val_edges))
        
        X_val_neg = extract_features(G_train, val_non_edges)
        y_val_neg = np.zeros(len(val_non_edges))
        
        X_val = np.vstack((X_val, X_val_neg))
        y_val = np.hstack((y_val, y_val_neg))
        
        # Use community information to enhance features
        print("Enhancing features with community information...")
        community_features_train = self._enhance_with_community_features(train_edges + train_non_edges)
        community_features_val = self._enhance_with_community_features(val_edges + val_non_edges)
        
        X_train = np.hstack((X_train, community_features_train))
        X_val = np.hstack((X_val, community_features_val))
        
        # Update PSO dimensions based on new feature count
        self.pso_params['num_dimensions'] = X_train.shape[1]
        
        # Initialize PSO with correct dimensions
        self.pso = PSO(**self.pso_params)
        
        print("Phase 3: Running PSO to optimize feature weights...")
        self.optimal_weights, best_fitness = self.pso.optimize(X_train, y_train, X_val, y_val)
        
        print(f"Optimization complete. Best AUC: {best_fitness:.4f}")
        print(f"Optimal weights: {self.optimal_weights}")
        
        return self.optimal_weights, best_fitness
        
    # Other methods remain the same...
    def predict(self, G_test, node_pairs):
        """
        Predict link probabilities for the given node pairs
        
        Args:
            G_test: Graph to extract features from
            node_pairs: List of node pairs to predict
            
        Returns:
            probabilities: Link prediction probabilities for each node pair
        """
        # Extract features for test pairs
        X_test = extract_features(G_test, node_pairs)
        
        # Enhance with community features
        community_features = self._enhance_with_community_features(node_pairs)
        X_test = np.hstack((X_test, community_features))
        
        # Calculate weighted scores
        scores = X_test.dot(self.optimal_weights)
        
        # Apply sigmoid to get probabilities
        probabilities = 1 / (1 + np.exp(-scores))
        
        return probabilities
def k_fold_cross_validation(G, k=5, random_state=42):
    """
    Perform k-fold cross-validation on the graph data
    
    Args:
        G: Input graph
        k: Number of folds
        random_state: Random seed for reproducibility
        
    Returns:
        mean_auc: Mean AUC across folds
        mean_ap: Mean average precision across folds
        std_auc: Standard deviation of AUC
        std_ap: Standard deviation of average precision
    """
    print(f"Performing {k}-fold cross-validation...")
    
    # Get all edges and non-edges
    all_edges = list(G.edges())
    all_non_edges = list(nx.non_edges(G))
    
    # Sample same number of non-edges as edges for balanced dataset
    random.seed(random_state)
    sampled_non_edges = random.sample(all_non_edges, len(all_edges))
    
    # Combine edges and non-edges
    all_pairs = all_edges + sampled_non_edges
    all_labels = np.concatenate([np.ones(len(all_edges)), np.zeros(len(sampled_non_edges))])
    
    # Prepare K-fold
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
    fold_aucs = []
    fold_aps = []
    fold_weights = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(all_pairs)):
        print(f"\nFold {fold_idx+1}/{k}")
        
        # Get train and test pairs
        train_pairs = [all_pairs[i] for i in train_idx]
        test_pairs = [all_pairs[i] for i in test_idx]
        train_labels = all_labels[train_idx]
        test_labels = all_labels[test_idx]
        
        # Split train pairs into edges and non-edges
        train_edges = []
        train_non_edges = []
        
        for i, pair in enumerate(train_pairs):
            if train_labels[i] == 1:
                train_edges.append(pair)
            else:
                train_non_edges.append(pair)
        
        # Create training graph (only with training edges)
        G_train = nx.Graph()
        G_train.add_nodes_from(G.nodes())
        G_train.add_edges_from(train_edges)
        
        # Initialize and run hybrid model on this fold
        hybrid_model = ACO_PSO_Hybrid(
            G_train,
            aco_params={
                'num_ants': 30,
                'num_iterations': 20,  # Reduced for cross-validation
                'alpha': 1.0,
                'beta': 2.0,
                'evaporation_rate': 0.5,
                'Q': 100
            },
            pso_params={
                'num_particles': 20,
                'max_iter': 30,  # Reduced for cross-validation
                'w': 0.7,
                'c1': 1.5,
                'c2': 1.5
            }
        )
        
        # Train the model
        optimal_weights, _ = hybrid_model.run(validation_size=0.2)
        fold_weights.append(optimal_weights)
        
        # Get predictions for test set
        predictions = hybrid_model.predict(G_train, test_pairs)
        
        # Calculate metrics
        auc = roc_auc_score(test_labels, predictions)
        ap = average_precision_score(test_labels, predictions)
        
        fold_aucs.append(auc)
        fold_aps.append(ap)
        
        print(f"Fold {fold_idx+1} AUC: {auc:.4f}, AP: {ap:.4f}")
    
    # Calculate mean and std of metrics
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    mean_ap = np.mean(fold_aps)
    std_ap = np.std(fold_aps)
    
    # Calculate average weights across folds
    avg_weights = np.mean(fold_weights, axis=0)
    
    print("\nCross-validation results:")
    print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Mean AP: {mean_ap:.4f} ± {std_ap:.4f}")
    
    # Print feature importance based on average weights
    feature_names = [
        "Common Neighbors", "Jaccard", "Adamic-Adar", 
        "Preferential Attachment", "Resource Allocation",
        "Same Community", "Pheromone Level"
    ]
    
    print("\nAverage Feature Importance:")
    for i, weight in enumerate(avg_weights):
        if i < len(feature_names):
            print(f"{feature_names[i]}: {weight:.4f}")
        else:
            print(f"Feature {i}: {weight:.4f}")
    
    return mean_auc, mean_ap, std_auc, std_ap, avg_weights

def main():
    # Load data
    print("Loading Facebook dataset...")
    G = load_facebook_data("facebook_combined.txt")
    print(f"Dataset loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Perform k-fold cross-validation
    mean_auc, mean_ap, std_auc, std_ap, avg_weights = k_fold_cross_validation(G, k=5)
    
    # For comparison, also run the original method once
    print("\nRunning single train-test split for comparison...")
    G_train, test_edges, test_non_edges = split_graph(G, test_size=0.2)
    
    # Initialize and run hybrid model
    hybrid_model = ACO_PSO_Hybrid(
        G_train,
        aco_params={
            'num_ants': 50,
            'num_iterations': 30,
            'alpha': 1.0,
            'beta': 2.0,
            'evaporation_rate': 0.5,
            'Q': 100
        },
        pso_params={
            'num_particles': 30,
            'max_iter': 50,
            'w': 0.7,
            'c1': 1.5,
            'c2': 1.5
        }
    )
    
    # Run optimization
    optimal_weights, best_fitness = hybrid_model.run()
    
    # Evaluate on test set
    test_pairs = test_edges + test_non_edges
    test_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_non_edges))])
    
    # Get predictions
    predictions = hybrid_model.predict(G_train, test_pairs)
    
    # Calculate metrics
    auc = roc_auc_score(test_labels, predictions)
    ap = average_precision_score(test_labels, predictions)
    
    print("\nSingle split results:")
    print(f"Test AUC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    
    # Compare with cross-validation
    print("\nComparison:")
    print(f"Single split - AUC: {auc:.4f}, AP: {ap:.4f}")
    print(f"Cross-validation - AUC: {mean_auc:.4f} ± {std_auc:.4f}, AP: {mean_ap:.4f} ± {std_ap:.4f}")
    
if __name__ == "__main__":
    main()