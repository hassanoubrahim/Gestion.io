import matplotlib.pyplot as plt
import networkx as nx

# Create a graph from the correlation groups
G = nx.Graph()

# Add edges to the graph based on correlation groups
for correlation_value, pairs in correlation_groups.items():
    for pair in pairs:
        G.add_edge(pair[0], pair[1], weight=correlation_value)

# Set the positions using spring layout for better visualization
pos = nx.spring_layout(G, k=0.15, iterations=20)

# Draw the graph
plt.figure(figsize=(15, 10))
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', edgecolors='black')
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7)
nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

# Draw edge labels with correlation values
edge_labels = {(u, v): f'{d["weight"]:.10f}' for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title("Correlation Graph of Mutations")
plt.show()


# Find pairs of features with high correlation (e.g., > 0.99)
high_correlation_pairs = correlation_matrix[(correlation_matrix > 0.99) & (correlation_matrix < 1.0)]

# Group features with the same correlation number
correlation_groups = {}
for col in high_correlation_pairs.columns:
    for row in high_correlation_pairs.index:
        if pd.notna(high_correlation_pairs.loc[row, col]):
            correlation_value = high_correlation_pairs.loc[row, col]
            if correlation_value not in correlation_groups:
                correlation_groups[correlation_value] = []
            correlation_groups[correlation_value].append((row, col))

# Display the correlation groups
correlation_groups
