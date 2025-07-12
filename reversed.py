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
            aa_index += 1 / np.log(G.degree(neighbor) + 1e-10)  # Added small constant to avoid div by zero
        
        # Preferential Attachment
        pref_attachment = G.degree(u) * G.degree(v)
        
        # Resource Allocation
        ra_index = 0
        for neighbor in nx.common_neighbors(G, u, v):
            ra_index += 1 / G.degree(neighbor)
            
        features.append([common_neighbors, jaccard, aa_index, pref_attachment, ra_index])
    
    return np.array(features)

class PSO_Community:
    """
    Particle Swarm Optimization for community detection in social networks
    """
    def __init__(self, G, num_particles=30, max_iter=100, num_communities=10, w=0.7, c1=1.5, c2=1.5):
        self.G = G
        self.num_particles = num_particles
        self.num_nodes = G.number_of_nodes()
        self.nodes = list(G.nodes())
        self.max_iter = max_iter
        self.num_communities = num_communities
        self.w = w          # Inertia weight
        self.c1 = c1        # Cognitive coefficient
        self.c2 = c2        # Social coefficient
        
        # For each node, assign a community label (between 0 and num_communities-1)
        # Each particle is a potential solution (community assignment)
        self.particles = np.random.randint(0, self.num_communities, size=(num_particles, self.num_nodes))
        self.velocities = np.random.rand(num_particles, self.num_nodes) * 0.1
        
        # Initialize best positions
        self.pbest = self.particles.copy()
        self.pbest_fitness = np.zeros(num_particles)
        
        # Initialize global best
        self.gbest = np.zeros(self.num_nodes, dtype=int)
        self.gbest_fitness = -np.inf
        
        # Node mapping from index to node ID
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(self.nodes)}
        
        # Final communities
        self.communities = []
    
    def fitness_function(self, community_assignment):
        """
        Calculate modularity as the fitness function
        
        Args:
            community_assignment: An array where the index is the node's index and the value is the community label
            
        Returns:
            modularity score
        """
        # Convert to dict of sets for efficiency
        communities = {}
        for i, comm_id in enumerate(community_assignment):
            node = self.idx_to_node[i]
            if comm_id not in communities:
                communities[comm_id] = set()
            communities[comm_id].add(node)
        
        # Calculate modularity
        m = self.G.number_of_edges()
        if m == 0:
            return 0
            
        modularity = 0
        for community in communities.values():
            for u in community:
                for v in community:
                    if u == v:
                        continue
                    
                    # Actual adjacency term (A_ij)
                    if self.G.has_edge(u, v):
                        actual = 1
                    else:
                        actual = 0
                    
                    # Expected term (k_i * k_j / 2m)
                    expected = self.G.degree(u) * self.G.degree(v) / (2 * m)
                    
                    modularity += (actual - expected)
        
        modularity /= (2 * m)
        return modularity
    
    def update_position(self, particle, velocity):
        """
        Update particle position based on velocity
        For community detection, we need to map continuous velocity to discrete community assignments
        """
        # Apply velocity to get continuous position
        continuous_pos = particle.astype(float) + velocity
        
        # Map to discrete community assignments using probabilistic mapping
        # Higher value = higher chance of being assigned to higher community number
        new_position = np.zeros_like(particle)
        
        for i in range(len(continuous_pos)):
            # Sigmoid function to map to [0,1]
            probs = np.zeros(self.num_communities)
            for j in range(self.num_communities):
                # Calculate probability for each community
                diff = abs(continuous_pos[i] - j)
                probs[j] = 1 / (1 + np.exp(diff))
            
            # Normalize probabilities
            probs /= np.sum(probs)
            
            # Select community based on probability
            new_position[i] = np.random.choice(self.num_communities, p=probs)
        
        return new_position.astype(int)
    
    def run(self):
        """Run PSO algorithm to find communities"""
        print("Running PSO for community detection...")
        
        # Initialize pbest_fitness values
        for i in range(self.num_particles):
            self.pbest_fitness[i] = self.fitness_function(self.particles[i])
        
        # Find initial gbest
        best_idx = np.argmax(self.pbest_fitness)
        self.gbest = self.pbest[best_idx].copy()
        self.gbest_fitness = self.pbest_fitness[best_idx]
        
        # Main PSO loop
        for iteration in range(self.max_iter):
            # Update velocities and positions
            for i in range(self.num_particles):
                # Random coefficients
                r1 = np.random.rand(self.num_nodes)
                r2 = np.random.rand(self.num_nodes)
                
                # Update velocity
                cognitive = self.c1 * r1 * (self.pbest[i] - self.particles[i])
                social = self.c2 * r2 * (self.gbest - self.particles[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                
                # Update position (discrete mapping)
                self.particles[i] = self.update_position(self.particles[i], self.velocities[i])
                
                # Evaluate new position
                fitness = self.fitness_function(self.particles[i])
                
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
                print(f"Iteration {iteration}, Best modularity: {self.gbest_fitness:.4f}")
        
        # Extract final communities
        self._extract_communities()
        
        return self.communities, self.gbest_fitness
    
    def _extract_communities(self):
        """Extract communities from the best solution"""
        community_dict = {}
        for i, comm_id in enumerate(self.gbest):
            node = self.idx_to_node[i]
            if comm_id not in community_dict:
                community_dict[comm_id] = set()
            community_dict[comm_id].add(node)
        
        # Convert dict of sets to list of sets
        self.communities = [comm for comm in community_dict.values() if len(comm) > 0]
        
        # Calculate edge density within each community
        densities = []
        for community in self.communities:
            if len(community) <= 1:
                densities.append(0)
                continue
                
            possible_edges = len(community) * (len(community) - 1) / 2
            actual_edges = 0
            for u in community:
                for v in community:
                    if u < v and self.G.has_edge(u, v):
                        actual_edges += 1
            
            density = actual_edges / possible_edges if possible_edges > 0 else 0
            densities.append(density)
        
        print(f"Detected {len(self.communities)} communities")
        print(f"Average community density: {np.mean(densities):.4f}")
        
        return self.communities


class ACO_FeatureOptimizer:
    """
    Ant Colony Optimization for feature weight optimization in link prediction
    """
    def __init__(self, num_ants=50, num_iterations=100, num_features=5, alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.num_features = num_features
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        
        # Initialize pheromone trails for each feature weight
        # Each feature's weight can range from 0 to 1, discretized into 10 bins
        self.weight_bins = 10
        self.pheromone = np.ones((num_features, self.weight_bins))
        
        # Initialize best solution
        self.best_weights = np.zeros(num_features)
        self.best_fitness = -np.inf
    
    def _select_weight(self, feature_idx, heuristic_info=None):
        """
        Select a weight for a feature based on pheromone levels
        
        Args:
            feature_idx: Index of the feature
            heuristic_info: Optional heuristic information to guide selection
            
        Returns:
            selected weight value (between 0 and 1)
        """
        weights = np.linspace(0, 1, self.weight_bins)
        
        # Calculate probabilities based on pheromone
        pheromone_values = self.pheromone[feature_idx]
        
        if heuristic_info is not None:
            heuristic_values = heuristic_info[feature_idx]
            attractiveness = (pheromone_values ** self.alpha) * (heuristic_values ** self.beta)
        else:
            attractiveness = pheromone_values ** self.alpha
        
        # Calculate probabilities
        total = np.sum(attractiveness)
        if total == 0:
            probabilities = np.ones(self.weight_bins) / self.weight_bins
        else:
            probabilities = attractiveness / total
        
        # Select bin
        selected_bin = np.random.choice(self.weight_bins, p=probabilities)
        
        # Add some noise for exploration
        selected_weight = weights[selected_bin] + np.random.uniform(-0.05, 0.05)
        selected_weight = max(0, min(1, selected_weight))  # Clip to [0, 1]
        
        return selected_weight
    
    def _construct_solution(self, heuristic_info=None):
        """
        Construct a solution (feature weights) for one ant
        
        Returns:
            weights: Array of feature weights
        """
        weights = np.zeros(self.num_features)
        
        for i in range(self.num_features):
            weights[i] = self._select_weight(i, heuristic_info)
        
        return weights
    
    def _update_pheromones(self, solutions, fitness_values):
        """
        Update pheromone trails based on solution quality
        
        Args:
            solutions: List of weight arrays
            fitness_values: Corresponding fitness values
        """
        # Evaporation
        self.pheromone *= (1 - self.evaporation_rate)
        
        # Deposit new pheromones
        for i, solution in enumerate(solutions):
            fitness = fitness_values[i]
            
            # Skip if fitness is bad
            if fitness <= 0:
                continue
            
            # Determine bin for each weight
            for feat_idx, weight in enumerate(solution):
                bin_idx = min(int(weight * self.weight_bins), self.weight_bins - 1)
                
                # Deposit pheromone proportional to fitness
                self.pheromone[feat_idx, bin_idx] += self.Q * fitness
    
    def fit(self, X_train, y_train, X_val, y_val, heuristic_info=None):
        """
        Optimize feature weights using ACO
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            heuristic_info: Optional heuristic information for each feature
            
        Returns:
            best_weights: Optimized feature weights
            best_fitness: Best fitness value achieved
        """
        print("Running ACO for feature weight optimization...")
        
        for iteration in range(self.num_iterations):
            # Each ant constructs a solution
            solutions = []
            fitness_values = []
            
            for ant in range(self.num_ants):
                # Construct solution (feature weights)
                weights = self._construct_solution(heuristic_info)
                solutions.append(weights)
                
                # Evaluate solution
                fitness = self._fitness_function(weights, X_train, y_train, X_val, y_val)
                fitness_values.append(fitness)
                
                # Update best solution if better
                if fitness > self.best_fitness:
                    self.best_weights = weights.copy()
                    self.best_fitness = fitness
            
            # Update pheromones
            self._update_pheromones(solutions, fitness_values)
            
            # Print progress every 10 iterations
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Best fitness: {self.best_fitness:.4f}")
        
        print(f"Optimization complete. Best weights: {self.best_weights}")
        return self.best_weights, self.best_fitness
    
    def _fitness_function(self, weights, X_train, y_train, X_val, y_val):
        """
        Calculate fitness (AUC score) for a set of weights
        
        Args:
            weights: Feature weights
            X_train, y_train, X_val, y_val: Training and validation data
            
        Returns:
            auc_score: Area under ROC curve
        """
        # Calculate weighted scores
        val_scores = X_val.dot(weights)
        
        # Apply sigmoid to get probabilities
        val_scores = 1 / (1 + np.exp(-val_scores))
        
        # Calculate AUC
        try:
            auc_score = roc_auc_score(y_val, val_scores)
            return auc_score
        except:
            # Return low fitness if there's an error
            return 0.0


class PSO_ACO_Hybrid:
    """
    Hybrid model using PSO for community detection and ACO for feature weight optimization
    """
    def __init__(self, G, pso_params=None, aco_params=None):
        self.G = G
        
        # Default parameters if none provided
        if pso_params is None:
            pso_params = {
                'num_particles': 30,
                'max_iter': 50,
                'num_communities': max(10, int(G.number_of_nodes() / 50)),  # Adaptive community count
                'w': 0.7,
                'c1': 1.5,
                'c2': 1.5
            }
        
        if aco_params is None:
            aco_params = {
                'num_ants': 50,
                'num_iterations': 50,
                'alpha': 1.0,
                'beta': 2.0,
                'evaporation_rate': 0.5,
                'Q': 100
            }
            
        self.pso_params = pso_params
        self.aco_params = aco_params
        
        # Store communities and optimal weights
        self.communities = None
        self.optimal_weights = None
        
        # Initialize PSO for community detection
        self.pso = PSO_Community(G, **pso_params)
        
        # We'll initialize ACO later after we know the feature count
        self.aco = None
    
    def _enhance_with_community_features(self, node_pairs):
        """
        Add community-based features to node pairs
        
        Args:
            node_pairs: List of node pairs
            
        Returns:
            community_features: Array of community-based features
        """
        # Add community-based features
        features = np.zeros((len(node_pairs), 1))
        
        for i, (u, v) in enumerate(node_pairs):
            # Feature: Are nodes in same community?
            same_community = 0
            for community in self.communities:
                if u in community and v in community:
                    same_community = 1
                    break
            
            features[i] = [same_community]
        
        return features
    
    def run(self, validation_size=0.3, random_state=42):
        print("Phase 1: Running PSO to detect communities...")
        self.communities, modularity = self.pso.run()
        
        print(f"Detected {len(self.communities)} communities with modularity {modularity:.4f}")
        
        print("Phase 2: Preparing features for ACO optimization...")
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
        
        # Update ACO parameters based on feature count
        self.aco_params['num_features'] = X_train.shape[1]
        
        # Initialize ACO with correct feature count
        self.aco = ACO_FeatureOptimizer(**self.aco_params)
        
        print("Phase 3: Running ACO to optimize feature weights...")
        
        # Optional: Calculate feature importance heuristic
        # Simple correlation-based heuristic to guide ACO
        heuristic_info = []
        for i in range(X_train.shape[1]):
            corr = np.abs(np.corrcoef(X_train[:, i], y_train)[0, 1])
            heuristic_info.append([corr] * self.aco.weight_bins)
        heuristic_info = np.array(heuristic_info)
        
        # Run ACO optimization
        self.optimal_weights, best_fitness = self.aco.fit(X_train, y_train, X_val, y_val, heuristic_info)
        
        print(f"Optimization complete. Best AUC: {best_fitness:.4f}")
        print(f"Optimal weights: {self.optimal_weights}")
        
        return self.optimal_weights, best_fitness
    
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
        hybrid_model = PSO_ACO_Hybrid(
            G_train,
            pso_params={
                'num_particles': 20,
                'max_iter': 20,  # Reduced for cross-validation
                'num_communities': max(5, int(G.number_of_nodes() / 100)),
                'w': 0.7,
                'c1': 1.5,
                'c2': 1.5
            },
            aco_params={
                'num_ants': 30,
                'num_iterations': 20,  # Reduced for cross-validation
                'alpha': 1.0,
                'beta': 2.0,
                'evaporation_rate': 0.5,
                'Q': 100
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
        "Same Community"
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
    
    # Run full model on single train-test split
    print("\nRunning single train-test split for comparison...")
    G_train, test_edges, test_non_edges = split_graph(G, test_size=0.2)
    
    # Initialize and run hybrid model
    hybrid_model = PSO_ACO_Hybrid(
        G_train,
        pso_params={
            'num_particles': 30,
            'max_iter': 30,
            'num_communities': max(10, int(G.number_of_nodes() / 50)),
            'w': 0.7,
            'c1': 1.5,
            'c2': 1.5
        },
        aco_params={
            'num_ants': 40,
            'num_iterations': 40,
            'alpha': 1.0,
            'beta': 2.0,
            'evaporation_rate': 0.5,
            'Q': 100
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
    print(f"Cross-Validation AUC: Mean={mean_auc:.4f}, Std={std_auc:.4f}")
    print(f"Cross-Validation AP:  Mean={mean_ap:.4f}, Std={std_ap:.4f}")
    print(f"Single Split AUC:     {auc:.4f}")
    print(f"Single Split AP:      {ap:.4f}")

if __name__ == "__main__":
    main()