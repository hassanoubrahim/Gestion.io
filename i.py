%%time
import shap

# Define the number of samples to summarize the background
K = 5  # Reduce K for faster computation

# Use shap.sample to summarize the background
background_summary = shap.sample(X_train, K)

# Calculate SHAP values with the summarized background
explainer = shap.KernelExplainer(svm_model.predict_proba, background_summary, link="logit")

# Parallelize computation of SHAP values
shap_values = explainer.shap_values(X_test, n_jobs=-1)

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt

# Example: Load your data
_df = df_merged#.replace(0, -1)  # [df_merged["Drug"] != 'Sensitive']
_df = _df[_df["Drug"] != 'Other *']
#_df = _df[_df["Drug"] != 'Mono']
_df = _df[_df["Drug"] != 'Sensitive']
#X = _df[sorted_grouped.index.to_list()]  # .replace(np.nan, 0)
X = _df[np.random.choice(df.columns, 4000)]  # .replace(np.nan, 0)
y = _df["Drug"].map({'Mono': 0, 'MDR': 1, 'Pre-XDR': 2, 'Sensitive': 3})

print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define oversampling strategy
oversample = RandomOverSampler()
X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)

# Scale the data
scaler = StandardScaler()
#X_train_resampled = scaler.fit_transform(X_train_resampled)
#X_test = scaler.transform(X_test)

# Build the model
logistic_model = LogisticRegression(random_state=42)

# Apply cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(logistic_model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')
print(f'Cross-validation accuracy scores: {cv_scores}')
print(f'Mean cross-validation accuracy: {cv_scores.mean()}')

# Train the model on the entire training set
logistic_model.fit(X_train_resampled, y_train_resampled)

# Evaluate the model
y_pred = logistic_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_accuracy}")

# Generate classification report
print(classification_report(y_test, y_pred))

# Calculate AUC-ROC for each class
y_pred_proba = logistic_model.predict_proba(X_test)
for i in range(len(logistic_model.classes_)):
    roc_auc = roc_auc_score(y_test == i, y_pred_proba[:, i])
    print(f"Class {i} AUC-ROC: {roc_auc}")

# Plot ROC curve for each class
fpr = dict()
tpr = dict()
for i in range(len(logistic_model.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_proba[:, i])
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc_score(y_test == i, y_pred_proba[:, i]):.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC Curve')
plt.legend(loc="lower right")
plt.show()
