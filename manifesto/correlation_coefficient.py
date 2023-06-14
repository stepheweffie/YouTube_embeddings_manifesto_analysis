import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Assuming df1 and df2 are pandas DataFrames with the same shape
assert df1.shape == df2.shape

num_embeddings = df1.shape[0]

# Empty list to hold correlation coefficients
correlations = []

# Loop through each pair of embeddings
for i in range(num_embeddings):
    embedding1 = df1.iloc[i].values
    embedding2 = df2.iloc[i].values

    # Compute correlation and p-value
    correlation, _ = pearsonr(embedding1, embedding2)
    correlations.append(correlation)

# Plotting

# 1. Histogram
plt.figure(figsize=(8, 6))
plt.hist(correlations, bins=50)
plt.title('Histogram of Correlation Coefficients')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Frequency')
plt.show()


# 3. Line plot
plt.figure(figsize=(8, 6))
plt.plot(correlations)
plt.title('Line Plot of Correlation Coefficients')
plt.xlabel('Pair Index')
plt.ylabel('Correlation Coefficient')
plt.show()
