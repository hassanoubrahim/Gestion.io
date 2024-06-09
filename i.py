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
