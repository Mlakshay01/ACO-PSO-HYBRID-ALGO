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

class ACO_LinkPredictor:
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
        self.communities = []
        
        # Feature weights (will be learned from pheromone distributions)
        self.weights = np.ones(7) / 7  # Equal weights initially for 7 features
    
    def run(self):
        print("Running ACO for link prediction...")
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
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Best path length: {len(best_path)}")
        
        # Extract communities and patterns from solutions
        self._extract_communities(best_solutions)
        
        # Learn weights from pheromone distribution
        self._learn_weights()
        
        return self.communities, self.pheromone, self.weights
    
    def _construct_solution(self):
        # Start from a random node
        current_node = int(random.choice(self.nodes))
        path = [current_node]
        visited = {current_node}
        
        # Construct path
        for _ in range(random.randint(2, 10)):  # Random path length
            candidates = []
            probabilities = []
            
            # Calculate probabilities for next nodes
            for neighbor in self.G.neighbors(current_node):
                if neighbor not in visited:
                    edge = (min(current_node, neighbor), max(current_node, neighbor))
                    
                    # Calculate attractiveness based on pheromone and degree
                    pheromone = self.pheromone.get(edge, 1.0)
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
                u, v = int(path[i]), int(path[i+1])
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
        
        print(f"Detected {len(self.communities)} communities")
        return self.communities
    
    def _learn_weights(self):
        # Calculate edge strengths based on pheromone levels
        edge_strengths = np.array(list(self.pheromone.values()))
        
        # Normalize pheromone levels
        max_pheromone = np.max(edge_strengths)
        if max_pheromone > 0:
            edge_strengths = edge_strengths / max_pheromone
        
        # Get high pheromone edges (likely links)
        high_pheromone_edges = []
        low_pheromone_edges = []
        
        # Sort edges by pheromone level
        edges_sorted = sorted(self.pheromone.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 10% as high pheromone and bottom 10% as low pheromone
        num_top = int(len(edges_sorted) * 0.1)
        high_pheromone_edges = [edge for edge, _ in edges_sorted[:num_top]]
        low_pheromone_edges = [edge for edge, _ in edges_sorted[-num_top:]]
        
        # Extract features for these edges
        X_high = extract_features(self.G, high_pheromone_edges)
        X_low = extract_features(self.G, low_pheromone_edges)
        
        # Add community features
        X_high_community = self._enhance_with_community_features(high_pheromone_edges)
        X_low_community = self._enhance_with_community_features(low_pheromone_edges)
        
        X_high = np.hstack((X_high, X_high_community))
        X_low = np.hstack((X_low, X_low_community))
        
        # Calculate feature differences between high and low pheromone edges
        feature_diffs = np.mean(X_high, axis=0) - np.mean(X_low, axis=0)
        
        # Convert to weights
        weights = np.abs(feature_diffs)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        self.weights = weights
        
        # Print feature importance
        feature_names = [
            "Common Neighbors", "Jaccard", "Adamic-Adar", 
            "Preferential Attachment", "Resource Allocation",
            "Same Community", "Pheromone Level"
        ]
        
        print("\nFeature Importance:")
        for i, weight in enumerate(self.weights):
            if i < len(feature_names):
                print(f"{feature_names[i]}: {weight:.4f}")
            else:
                print(f"Feature {i}: {weight:.4f}")
                
        return self.weights
    
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
            pheromone = self.pheromone.get((min(u, v), max(u, v)), 0)
            
            features[i] = [same_community, pheromone]
        
        return features
    
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
        scores = X_test.dot(self.weights)
        
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
        
        # Initialize and run ACO model on this fold
        aco_model = ACO_LinkPredictor(
            G_train,
            num_ants=30,
            num_iterations=20,  # Reduced for cross-validation
            alpha=1.0,
            beta=2.0,
            evaporation_rate=0.5,
            Q=100
        )
        
        # Train the model
        _, _, _ = aco_model.run()
        
        # Get predictions for test set
        predictions = aco_model.predict(G_train, test_pairs)
        
        # Calculate metrics
        auc = roc_auc_score(test_labels, predictions)
        ap = average_precision_score(test_labels, predictions)
        
        fold_aucs.append(auc)
        fold_aps.append(ap)
        
        print(f"Fold {fold_idx+1} AUC: {auc:.4f}, AP: {ap:.4f}")
    
    # Calculate mean of metrics
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    mean_ap = np.mean(fold_aps)
    std_ap = np.std(fold_aps)
    
    print("\nCross-validation results:")
    print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Mean AP: {mean_ap:.4f} ± {std_ap:.4f}")
    
    return mean_auc, mean_ap, std_auc, std_ap

def main():
    # Load data
    print("Loading Facebook dataset...")
    G = load_facebook_data("facebook_combined.txt")
    print(f"Dataset loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Perform k-fold cross-validation
    mean_auc, mean_ap, std_auc, std_ap = k_fold_cross_validation(G, k=5)
    
    # Run on a single train-test split
    print("\nRunning single train-test split evaluation...")
    G_train, test_edges, test_non_edges = split_graph(G, test_size=0.2)
    
    # Initialize and run ACO model
    aco_model = ACO_LinkPredictor(
        G_train,
        num_ants=50,
        num_iterations=50,
        alpha=1.0,
        beta=2.0,
        evaporation_rate=0.5,
        Q=100
    )
    
    # Run optimization
    _, _, _ = aco_model.run()
    
    # Evaluate on test set
    test_pairs = test_edges + test_non_edges
    test_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_non_edges))])
    
    # Get predictions
    predictions = aco_model.predict(G_train, test_pairs)
    
    # Calculate metrics
    auc = roc_auc_score(test_labels, predictions)
    ap = average_precision_score(test_labels, predictions)
    
    print("\nSingle split results:")
    print(f"Test AUC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    
if __name__ == "__main__":
    main()