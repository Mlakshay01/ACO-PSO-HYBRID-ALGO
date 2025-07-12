import networkx as nx
from sklearn.model_selection import train_test_split

# Path to the input file (update if needed)
input_path = "/Users/tushar/Documents/Startups/minor/minor-phase-2/facebook_combined.txt"

# Load the graph from edge list
G = nx.read_edgelist(input_path, nodetype=int)

# Get all edges from the graph
edges = list(G.edges())

# Split edges into train and test (80-20 split)
train_edges, test_edges = train_test_split(edges, test_size=0.2, random_state=42)

# Write train edges to file
with open("/Users/tushar/Documents/Startups/minor/minor-phase-2/train_edges.txt", "w") as f_train:
    for u, v in train_edges:
        f_train.write(f"{u} {v}\n")

# Write test edges to file
with open("/Users/tushar/Documents/Startups/minor/minor-phase-2/test_edges.txt", "w") as f_test:
    for u, v in test_edges:
        f_test.write(f"{u} {v}\n")

print("✅ Split complete.")
print(f"🔹 Total edges: {len(edges)}")
print(f"🔹 Train edges: {len(train_edges)} (saved in train_edges.txt)")
print(f"🔹 Test edges: {len(test_edges)} (saved in test_edges.txt)")