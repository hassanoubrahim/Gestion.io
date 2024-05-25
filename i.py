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
