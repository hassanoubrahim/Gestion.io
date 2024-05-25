import networkx as nx

# Define the correlation threshold
threshold = 0.8

# Create an empty graph
G = nx.Graph()

# Add nodes and edges based on the correlation matrix
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) >= threshold:
            G.add_edge(correlation_matrix.columns[i], correlation_matrix.columns[j], weight=correlation_matrix.iloc[i, j])

# Draw the graph
plt.figure(figsize=(14, 12))
pos = nx.spring_layout(G, k=0.5)
edges = G.edges(data=True)
weights = [abs(edge[2]['weight']) for edge in edges]
nx.draw_networkx(G, pos, with_labels=True, node_size=7000, node_color='lightblue', font_size=10, font_weight='bold', edge_color=weights, edge_cmap=plt.cm.viridis, width=3)
plt.title('Network of Correlated Features')
plt.show()

# Define the correlation threshold
threshold = 0.8

# Create an empty directed graph
G = nx.DiGraph()

# Add nodes
for node in correlation_matrix.columns:
    G.add_node(node)

# Add edges based on closest correlation for each feature
for i in correlation_matrix.columns:
    closest_feature = None
    max_corr = threshold
    
    for j in correlation_matrix.columns:
        if i != j and abs(correlation_matrix.loc[i, j]) > max_corr:
            closest_feature = j
            max_corr = abs(correlation_matrix.loc[i, j])
    
    if closest_feature:
        G.add_edge(i, closest_feature, weight=correlation_matrix.loc[i, closest_feature])
