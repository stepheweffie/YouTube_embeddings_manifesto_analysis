from scipy.spatial import distance
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE, MDS
import umap
from manifesto.visuals.dimensionality_visuals import cosine_similarity_plot


data = pd.read_pickle(f'manifesto/data/PDFEmbeddingsTask/PDFEmbeddingsTask__99914b932b-data.pkl')
data = data['pdf_embeddings']

# PDF data is stored in the variable 'data'
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Detect

first_col = data.iloc[:, 0]  # iloc allows you to select by integer-based location
video_data = pd.read_pickle(f'manifesto/data/single_video_openai_embeddings.pkl')
df1 = pd.DataFrame(video_data)
df2 = pd.DataFrame(data)
distances = distance.cdist(df1.values, df2.values, 'euclidean')
correlation = df1.corrwith(df2, axis=1)

# Assuming df1 and df2 are your two dataframes containing BERT embeddings
similarity = cosine_similarity(df1.mean(axis=0).values.reshape(1, -1), df2.mean(axis=0).values.reshape(1, -1))
print(f"The cosine similarity between the two documents is {similarity[0][0]}")
# The cosine similarity between the two documents is 0.854043123080344

# fitting the MDS with n_components as 2
mds = MDS(n_components=2)
projected_distances = mds.fit_transform(distances)

plt.scatter(projected_distances[:, 0], projected_distances[:, 1])
plt.show()
# create a scaler object
scaler = StandardScaler()
# fit and transform the data
df1_normalized = pd.DataFrame(scaler.fit_transform(df1), columns = df1.columns)
# Do the same for df2
df2_normalized = pd.DataFrame(scaler.fit_transform(df2), columns = df2.columns)

# Assuming df1 and df2 are your dataframes, and that they are already normalized
# Let's assume df2 has more columns than df1
print(df1_normalized.shape[0], df2_normalized.shape[0])
# Let's assume df2 has more rows than df1
if df2_normalized.shape[0] > df1_normalized.shape[0]:
    # Calculate difference in number of rows
    diff = df2_normalized.shape[0] - df1_normalized.shape[0]
    # Create a new dataframe of zeros with 'diff' rows and the same number of columns as df1
    padding_df = pd.DataFrame(np.zeros((diff, df1_normalized.shape[1])))
    # Concatenate df1 with the padding dataframe along axis 0 (vertically)
    df1_padded = pd.concat([df1_normalized, padding_df], axis=0).reset_index(drop=True)
    cos_sim_plot = cosine_similarity_plot(df1_padded, df2_normalized)
    plt.show()

# Perform PCA
pca = PCA(n_components=2)
reduced_df1 = pca.fit_transform(df1_normalized)
reduced_df2 = pca.fit_transform(df2_normalized)
# print(pca.components_)

# Plot the results
plt.scatter(reduced_df1[:, 0], reduced_df1[:, 1], label='Video', alpha=0.5)
plt.scatter(reduced_df2[:, 0], reduced_df2[:, 1], label='Manifesto', alpha=0.5)
plt.legend()
plt.title('Video vs Manifesto Text Embeddings')  # Set the title of the plot
plt.show()

# Create a correlation matrix for each dataframe
corr_df1 = df1_normalized.corr()
corr_df2 = df2_normalized.corr()
reduced_df1 = pd.DataFrame(reduced_df1)
reduced_df2 = pd.DataFrame(reduced_df2)
# Assuming df1 and df2 are your PCA-transformed dataframes
data = pd.concat([reduced_df1, reduced_df2])

# Choose the number of clusters k (this is a hyperparameter you'll need to decide on)
k = 10  # or whatever you choose
# Fit a KMeans model to your data
kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
# Add the KMeans labels to your dataframe
data['cluster'] = kmeans.labels_
# Create a scatterplot of the first two principal components, colored by KMeans cluster label
for i in range(k):
    plt.scatter(data[data['cluster'] == i].iloc[:, 0], data[data['cluster'] == i].iloc[:, 1], label='Cluster ' + str(i))
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('KMeans Clustering of PCA-Transformed Data')
plt.legend()
plt.show()

# Assuming df1 and df2 are your PCA-transformed dataframes
data = pd.concat([reduced_df1, reduced_df2])
# Fit a DBSCAN model to your data
# Here, eps and min_samples are hyperparameters you may need to adjust based on your data
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(data)
# Add the DBSCAN cluster labels to your dataframe
# Note that points labeled as -1 are considered noise by DBSCAN
data['cluster'] = dbscan.labels_
# Create a scatterplot of the first two principal components, colored by DBSCAN cluster label
clusters = data['cluster'].unique()
for cluster in clusters:
    plt.scatter(data[data['cluster'] == cluster].iloc[:, 0], data[data['cluster'] == cluster].iloc[:, 1],
                label='Cluster ' + str(cluster))
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('DBSCAN Clustering of PCA-Transformed Data')
plt.legend()
plt.show()

# Pie chart of the explained variance ratio
plt.figure(figsize=(6, 4))
plt.pie(np.array([1,2]), pca.explained_variance_ratio_, labels=['PC1', 'PC2'])
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.tight_layout()
plt.show()

# Get the principal components
pca_components = pca.components_

# Display the contributions of original variables to PC1 and PC2
# print("PC1 contributions:", pca_components[0])
# print("PC2 contributions:", pca_components[1])
# Assuming pca is your trained PCA model

# Get the components
components = pca.components_
# For DataFrame df1
df1_columns = df1_normalized.columns
for i, component in enumerate(components):
    # print(f"Principal Component {i + 1}:")
    component_weights = dict(zip(df1_columns, component))
    sorted_weights = sorted(component_weights.items(), key=lambda x: x[1], reverse=True)
    # Display top contributing features for each component
    for feature, weight in sorted_weights[:10]:
        print(f"{feature}: {weight}")
    print("\n")

# Concatenate your dataframes into one, for easier plotting
concatenated_df = data

# Create a pairplot
sns.pairplot(data)
plt.show()


# First, you need to concatenate your two dataframes
data = pd.concat([df1_normalized, df2_normalized])
df1 = df1_normalized
# Next, we'll use t-SNE to reduce the dimensionality of your embeddings to 2 dimensions
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(data)

# Now we can plot the results. We'll use different colors for embeddings from df1 and df2
plt.figure(figsize=(6, 6))
plt.scatter(embeddings_2d[:len(df1), 0], embeddings_2d[:len(df1), 1], label='df1')
plt.scatter(embeddings_2d[len(df1):, 0], embeddings_2d[len(df1):, 1], label='df2')
plt.legend()
plt.show()


# Concatenate your two dataframes along the row axis
df = pd.concat([df1_normalized, df2_normalized], axis=0)

# Run UMAP on your combined dataframe
reducer = umap.UMAP()
embedding = reducer.fit_transform(df)

# Split the transformed data back into two parts
embedding1 = embedding[:len(df1_normalized)]
embedding2 = embedding[len(df1_normalized):]

# Plot the results
plt.scatter(embedding1[:, 0], embedding1[:, 1], label='df1', s=5)
plt.scatter(embedding2[:, 0], embedding2[:, 1], label='df2', s=5)
plt.legend()
plt.show()

