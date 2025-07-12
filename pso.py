import networkx as nx
import numpy as np
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

# Load the Facebook dataset
def load_facebook_data(file_path="facebook_combined.txt"):
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

class PSO_LinkPredictor:
    def __init__(self, num_particles=30, max_iter=100, w=0.7, c1=1.5, c2=1.5):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w          # Inertia weight
        self.c1 = c1        # Cognitive coefficient (personal best)
        self.c2 = c2        # Social coefficient (global best)
        
        # Number of dimensions will be set when features are extracted
        self.num_dimensions = None
        
        # Initialize particles
        self.particles = None
        self.velocities = None
        
        # Initialize best positions
        self.pbest = None
        self.pbest_fitness = None
        
        # Initialize global best
        self.gbest = None
        self.gbest_fitness = None
        
        # Final optimized weights
        self.optimal_weights = None
    
    def _initialize_particles(self, num_dimensions):
        """Initialize particles based on feature dimensions"""
        self.num_dimensions = num_dimensions
        
        # Initialize particles with random weights
        self.particles = np.random.rand(self.num_particles, self.num_dimensions)
        self.velocities = np.random.rand(self.num_particles, self.num_dimensions) * 0.1
        
        # Initialize best positions
        self.pbest = self.particles.copy()
        self.pbest_fitness = np.zeros(self.num_particles)
        
        # Initialize global best
        self.gbest = np.zeros(self.num_dimensions)
        self.gbest_fitness = -np.inf
    
    def fitness_function(self, weights, X_train, y_train, X_val, y_val):
        """
        Calculate fitness for a set of weights
        Fitness is AUC score on validation set
        """
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
        """
        Run PSO optimization to find optimal feature weights
        
        Args:
            X_train: Feature matrix for training set
            y_train: Labels for training set
            X_val: Feature matrix for validation set
            y_val: Labels for validation set
            
        Returns:
            optimal_weights: Optimized feature weights
            best_fitness: Best fitness score achieved
        """
        print("Running PSO optimization for link prediction...")
        
        # Initialize particles based on feature dimensions
        self._initialize_particles(X_train.shape[1])
        
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
        
        # Store optimal weights
        self.optimal_weights = self.gbest.copy()
        
        # Print feature importance based on weights
        feature_names = [
            "Common Neighbors", "Jaccard", "Adamic-Adar", 
            "Preferential Attachment", "Resource Allocation"
        ]
        
        print("\nFeature Importance:")
        for i, weight in enumerate(self.optimal_weights):
            if i < len(feature_names):
                print(f"{feature_names[i]}: {weight:.4f}")
            else:
                print(f"Feature {i}: {weight:.4f}")
        
        return self.optimal_weights, self.gbest_fitness
    
    def predict(self, X_test):
        """
        Predict link probabilities using optimized weights
        
        Args:
            X_test: Feature matrix for test instances
            
        Returns:
            probabilities: Link prediction probabilities
        """
        if self.optimal_weights is None:
            raise ValueError("Model not trained yet. Run optimize() first.")
        
        # Calculate weighted scores
        scores = X_test.dot(self.optimal_weights)
        
        # Apply sigmoid to get probabilities
        probabilities = 1 / (1 + np.exp(-scores))
        
        return probabilities

class PSO_LinkPrediction:
    def __init__(self, G, pso_params=None):
        self.G = G
        
        # Default parameters if none provided
        if pso_params is None:
            pso_params = {
                'num_particles': 30,
                'max_iter': 100,
                'w': 0.7,
                'c1': 1.5,
                'c2': 1.5
            }
            
        self.pso_params = pso_params
        self.pso = PSO_LinkPredictor(**pso_params)
        self.optimal_weights = None
        
        # Advanced feature engineering flags
        self.use_additional_features = True
        self.use_node_centrality = True
        
        # Store normalization parameters
        self.feature_means = None
        self.feature_stds = None
    
    def extract_advanced_features(self, G, node_pairs):
        """
        Extract advanced features beyond basic topological metrics
        
        Args:
            G: Input graph
            node_pairs: List of node pairs to extract features for
            
        Returns:
            features: Matrix of advanced features
        """
        if not self.use_additional_features:
            return np.zeros((len(node_pairs), 0))
        
        advanced_features = []
        
        # Pre-compute node centralities for efficiency
        if self.use_node_centrality:
            print("Computing node centrality measures...")
            betweenness = nx.betweenness_centrality(G)
            closeness = nx.closeness_centrality(G)
            pagerank = nx.pagerank(G)
        
        for u, v in node_pairs:
            features = []
            
            # Path-based measures
            try:
                shortest_path = nx.shortest_path_length(G, u, v)
                features.append(1.0 / shortest_path if shortest_path > 0 else 0)
            except nx.NetworkXNoPath:
                features.append(0)  # No path exists
            
            # Centrality-based measures
            if self.use_node_centrality:
                # Sum and product of centrality measures
                features.append(betweenness.get(u, 0) + betweenness.get(v, 0))
                features.append(closeness.get(u, 0) + closeness.get(v, 0))
                features.append(pagerank.get(u, 0) + pagerank.get(v, 0))
                
                # Centrality product (multiplication)
                features.append(betweenness.get(u, 0) * betweenness.get(v, 0))
                features.append(pagerank.get(u, 0) * pagerank.get(v, 0))
            
            advanced_features.append(features)
        
        return np.array(advanced_features)
        
    def run(self, validation_size=0.3, random_state=42):
        """
        Run PSO-based link prediction model
        
        Args:
            validation_size: Proportion of edges to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            optimal_weights: Optimized feature weights
            best_fitness: Best fitness score achieved
        """
        print("Preparing data for PSO optimization...")
        
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
        
        # Extract basic features
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
        
        # Extract and add advanced features if enabled
        if self.use_additional_features:
            print("Adding advanced features...")
            adv_train = self.extract_advanced_features(G_train, train_edges + train_non_edges)
            adv_val = self.extract_advanced_features(G_train, val_edges + val_non_edges)
            
            if adv_train.shape[1] > 0:  # Only if features were actually generated
                X_train = np.hstack((X_train, adv_train))
                X_val = np.hstack((X_val, adv_val))
        
        # Normalize features for better PSO performance
        feature_means = np.mean(X_train, axis=0)
        feature_stds = np.std(X_train, axis=0)
        
        # Replace zero std with 1 to avoid division by zero
        feature_stds[feature_stds == 0] = 1
        
        X_train = (X_train - feature_means) / feature_stds
        X_val = (X_val - feature_means) / feature_stds
        
        # Store normalization parameters for prediction
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        
        print(f"Feature matrix shape: {X_train.shape}")
        
        # Run PSO to optimize feature weights
        print("Running PSO optimization...")
        self.optimal_weights, best_fitness = self.pso.optimize(X_train, y_train, X_val, y_val)
        
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
        if self.optimal_weights is None:
            raise ValueError("Model not trained yet. Run the optimization first.")
            
        # Extract features for test pairs
        X_test = extract_features(G_test, node_pairs)
        
        # Add advanced features if used during training
        if self.use_additional_features:
            adv_test = self.extract_advanced_features(G_test, node_pairs)
            if adv_test.shape[1] > 0:
                X_test = np.hstack((X_test, adv_test))
        
        # Normalize features using stored parameters
        X_test = (X_test - self.feature_means) / self.feature_stds
        
        # Get predictions
        probabilities = self.pso.predict(X_test)
        
        return probabilities

def k_fold_cross_validation(G, k=5, random_state=42):
    """
    Perform k-fold cross-validation on the graph data using PSO
    
    Args:
        G: Input graph
        k: Number of folds
        random_state: Random seed for reproducibility
        
    Returns:
        mean_auc: Mean AUC across folds
        mean_ap: Mean average precision across folds
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
        
        # Initialize and run PSO model on this fold
        pso_model = PSO_LinkPrediction(
            G_train,
            pso_params={
                'num_particles': 20,
                'max_iter': 30,  # Reduced for cross-validation
                'w': 0.7,
                'c1': 1.5,
                'c2': 1.5
            }
        )
        
        # Set to use basic features only for faster cross-validation
        pso_model.use_additional_features = False
        pso_model.use_node_centrality = False
        
        # Train the model with smaller validation set
        optimal_weights, _ = pso_model.run(validation_size=0.2)
        fold_weights.append(optimal_weights)
        
        # Get predictions for test set
        predictions = pso_model.predict(G_train, test_pairs)
        
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
        "Preferential Attachment", "Resource Allocation"
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
    
    # Run on a single train-test split with full feature set
    print("\nRunning single train-test split evaluation with advanced features...")
    G_train, test_edges, test_non_edges = split_graph(G, test_size=0.2)
    
    # Initialize and run PSO model
    pso_model = PSO_LinkPrediction(
        G_train,
        pso_params={
            'num_particles': 30,
            'max_iter': 50,
            'w': 0.7,
            'c1': 1.5,
            'c2': 1.5
        }
    )
    
    # Enable advanced features for full model
    pso_model.use_additional_features = True
    pso_model.use_node_centrality = True
    
    # Run optimization
    optimal_weights, best_fitness = pso_model.run(validation_size=0.2)
    
    # Evaluate on test set
    test_pairs = test_edges + test_non_edges
    test_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_non_edges))])
    
    # Get predictions
    predictions = pso_model.predict(G_train, test_pairs)
    
    # Calculate metrics
    auc = roc_auc_score(test_labels, predictions)
    ap = average_precision_score(test_labels, predictions)
    
    print("\nSingle split results with advanced features:")
    print(f"Test AUC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    
    # Compare with cross-validation
    print("\nComparison:")
    print(f"Single split (advanced features) - AUC: {auc:.4f}, AP: {ap:.4f}")
    print(f"Cross-validation (basic features) - AUC: {mean_auc:.4f} ± {std_auc:.4f}, AP: {mean_ap:.4f} ± {std_ap:.4f}")

if __name__ == "__main__":
    main()