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

# Add directed edges between highly correlated nodes only
for i in corr_matrix.columns:
    for j in corr_matrix.index:
        if i != j and abs(corr_matrix.loc[i, j]) > threshold:
            # Directing edge from lower indexed feature to higher indexed feature for simplicity
            if corr_matrix.columns.get_loc(i) < corr_matrix.columns.get_loc(j):
                G.add_edge(i, j, weight=corr_matrix.loc[i, j])
