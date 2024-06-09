import matplotlib.pyplot as plt
# Get feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Select top N features
top_n = 10
top_indices = indices[:top_n]

# Plot the top N features
plt.figure(figsize=(10, 6))
plt.title("Top 10 Feature Importances")
plt.bar(range(top_n), importances[top_indices], align="center")
plt.xticks(range(top_n), [X.columns[i] for i in top_indices], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()
