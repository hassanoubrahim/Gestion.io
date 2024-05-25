import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('path_to_your_file.csv')

# Compute the correlation matrix
correlation_matrix = data.corr()

# Define the correlation threshold
threshold = 0.8

# Create an empty directed graph
G = nx.DiGraph()

# Initialize a set to keep track of visited nodes
visited = set()

# Add nodes and edges based on the closest correlation for each feature
for i in correlation_matrix.columns:
    if i not in visited:
        closest_feature = None
        max_corr = threshold
        
        for j in correlation_matrix.columns:
            if i != j and abs(correlation_matrix.loc[i, j]) > max_corr:
                closest_feature = j
                max_corr = abs(correlation_matrix.loc[i, j])
        
        if closest_feature:
            G.add_edge(i, closest_feature, weight=correlation_matrix.loc[i, closest_feature])
            visited.add(i)
            visited.add(closest_feature)

# Draw the graph
plt.figure(figsize=(14, 12))
pos = nx.spring_layout(G, k=0.5)
edges = G.edges(data=True)
weights = [abs(edge[2]['weight']) for edge in edges]
nx.draw_networkx(G, pos, with_labels=True, node_size=7000, node_color='lightblue', font_size=10, font_weight='bold', edge_color=weights, edge_cmap=plt.cm.viridis, width=3, arrowsize=20)
plt.title('Network of Closest Correlated Features')
plt.show()
