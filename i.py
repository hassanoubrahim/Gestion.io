import numpy as np
import matplotlib.pyplot as plt

# Get feature importances (coefficients)
feature_importance = svm_model.coef_[0]

# Rank features based on their coefficients
feature_ranks = np.argsort(np.abs(feature_importance))[::-1]

# Set a threshold (e.g., top 100 features)
threshold = 2335

# Select top features based on the threshold
top_features = feature_ranks[:threshold]
top_feature_names = X.columns[top_features]
top_feature_importance = feature_importance[top_features]

# Plot top feature importance
plt.figure(figsize=(12, 9))
plt.bar(top_feature_names, top_feature_importance)
plt.xlabel('Features')
plt.ylabel('Coefficient Magnitude')
plt.title('Top Feature Importance')
plt.xticks(rotation=90)
plt.show()

