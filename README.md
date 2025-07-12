# 🔗 Hybrid Link Prediction using ACO + PSO (and Reverse Hybrid)

This minor project presents a novel **hybrid and reverse-hybrid approach for link prediction** in graphs, combining:

- 🐜 **Ant Colony Optimization (ACO)** for simulation-based exploration of graph paths,
- 🐦 **Particle Swarm Optimization (PSO)** for supervised feature-weight learning,
- 🔄 A **hybrid ensemble** that leverages both pheromone trails and optimized topological features.

---

## 🎯 Project Objective

Predict **missing or future links** in social networks by building an intelligent, hybrid scoring system that:

- Learns the **topological structure** of graphs (PSO),
- Explores and ranks links through **probabilistic graph walks** (ACO),
- Combines both into an **ensemble link scorer**.

---

## 🧠 Key Contributions

| Model              | Description |
|--------------------|-------------|
| **ACO**            | Simulates biased random walks over the graph using pheromones. |
| **PSO**            | Optimizes a linear model over graph-based similarity metrics. |
| **Hybrid**         | Final score = `w₁ * PSO_score + w₂ * ACO_score` |
| **Reverse Hybrid** | Treats `ACO_score` as a sixth feature and re-optimizes using PSO. |

---

## 📁 Files

| File | Description |
|------|-------------|
| `minor.py` | ACO simulation and pheromone scoring of missing links |
| `pso.py` | PSO-based link predictor using topological features |
| `split.py` | Splits graph edges into training and testing sets |
| `train_edges.txt` | Output of edge-splitting process |
| `facebook_combined.txt` | *(Not included)* Dataset from SNAP: https://snap.stanford.edu/data/egonets-Facebook.html |

---

## 🔬 Features Used in PSO

1. Common Neighbors  
2. Jaccard Coefficient  
3. Adamic-Adar Index  
4. Preferential Attachment  
5. Resource Allocation  
6. *(Optional in Reverse Hybrid)* ACO pheromone score

---

## 🧪 Results

| Model             | AUC Score |
|-------------------|-----------|
| ACO Only          | *Qualitative* link ranking |
| PSO Only (5-feat) | ~0.983 |
| **Hybrid (weighted)** | **0.9919** ✅ |
| **Reverse Hybrid**    | **0.9904** |

> Hybrid model shows measurable improvement over PSO baseline by combining global graph-walk knowledge (ACO) and local topological structure (PSO).

---

## 🧠 How Hybrid Works

### 1. ACO Scoring (minor.py)
- Each ant walks the graph.
- Pheromones deposited on paths with repeated visits.
- Missing links are scored by their pheromone strength.

### 2. PSO Scoring (pso.py)
- Each link is represented as a feature vector.
- PSO learns the best weights using AUC as the fitness score.

### 3. Hybrid Scoring
- `Final_Score = 0.6 * PSO_Prediction + 0.4 * ACO_Pheromone`
- Links are ranked using this combined score.

### 4. Reverse Hybrid
- `ACO pheromone score` is added as a 6th feature into the PSO model.
- PSO retrains with 6 features → improved learning from pheromone dynamics.

---

## 🚀 How to Run

1. **Split Dataset**
```bash
python split.py
```

2. **Rn Hybrid and Reverse Hybrid Algo**
```bash
python hybrid.py
python reverse.py


