import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Example: Load your data
# X, y = load_your_data()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = Sequential()
model.add(Dense(1024, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_train.shape[1], activation='sigmoid'))  # Using 'sigmoid' for binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# Make predictions
y_pred = model.predict(X_test)

###################333
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
