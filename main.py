from scipy.spatial import distance
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, MDS
import umap
from manifesto.visuals.dimensionality_visuals import cosine_similarity_plot, pca_reduce_dimensionality, \
    pca_kmeans_dimensionality
from manifesto.visuals.correlation_matrices import networkx_graph
import vaex
import vaex.ml.cluster


# create a scaler object
scaler = StandardScaler()

data = pd.read_pickle(f'manifesto/data/single_video_openai_embeddings.pkl')
video_data = data
video_dataframe = pd.DataFrame(video_data)
df1_normalized = pd.DataFrame(scaler.fit_transform(video_dataframe))

# Do the same for df2
data = pd.read_pickle(f'manifesto/data/PDFEmbeddingsTask/PDFEmbeddingsTask__99914b932b-data.pkl')
pdf_data = data['pdf_embeddings']
pdf_dataframe = pd.DataFrame(pdf_data)
df2_normalized = pd.DataFrame(scaler.fit_transform(pdf_dataframe))

df1 = video_dataframe
df2 = pdf_dataframe

# text-embeddings-ada-002 cosine similarity histogram
cosine_similarity_plot(df1, df2)

# vaex array of scaled embeddings for PCA
vdf1 = vaex.from_pandas(df1_normalized)
vdf2 = vaex.from_pandas(df2_normalized)
# Remove NaN values
vdf1.dropna()
vdf2.dropna()


data_array = np.concatenate((vdf1, vdf2), axis=0)
ldf1 = len(vdf1)
ldf2 = len(data_array) - ldf1
data_array.sort()
print(data_array.shape)

data_array = vaex.from_arrays(x=vdf1, y=vdf2)
# Make sure both arrays are of the same length
# Initialize the KMeans model via vaex and fit the data
kmeans = vaex.ml.cluster.KMeans(features=['x', 'y'], n_clusters=10)
# Fit and transform the data
kmeans.fit(data_array)
df_trans = kmeans.transform(data_array)
df_trans['predicted_kmean_map'] = df_trans.prediction_kmeans.map(mapper={0: 1, 1: 2, 2: 0})
print(df_trans)

# PCA KMEANS dimensionality reduction
# pca_reduce_dimensionality(data_array, ldf1, ldf2)
# pca_kmeans_dimensionality(data_array)

# Scale a data array
data = pd.concat([df1, df2], axis=1)
data = scaler.fit_transform(data)

# Swarm plot
# sns.swarmplot(data, size=2)
# fit and transform the data for ada embeddings
# plt.show()

# KDE plot
# sns.kdeplot(data, shade=True)
# plt.show()


# Histogram of correlation coefficients
correlation = df1.corrwith(df2, axis=1)
plt.figure(figsize=(8, 6))
plt.hist(correlation, bins=50)
plt.title('Histogram of Correlation Coefficients')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Frequency')
plt.show()

# 2. Line plot of correlation coefficients
plt.figure(figsize=(8, 6))
plt.plot(correlation)
plt.title('Line Plot of Correlation Coefficients')
plt.xlabel('Pair Index')
plt.ylabel('Correlation Coefficient')
plt.show()

# Concatenate the two dataframes along the columns
df = pd.concat([df1, df2], axis=1)
# Calculate the correlation matrix
corr_matrix = df.corr()
# Networkx graph of the correlation matrix
networkx_graph(corr_matrix)
# networkx.algorithms.community.modularity_max.greedy_modularity_communities(corr_matrix)
# Plot the correlation matrices
# Assuming df1 and df2 are your two dataframes containing ada-002 embeddings
similarity = cosine_similarity(df1.mean(axis=0).values.reshape(1, -1), df2.mean(axis=0).values.reshape(1, -1))
print(f"The cosine similarity for OpenAI embeddings between the two documents is {similarity[0][0]}")
# The cosine similarity between the two documents is 0.854043123080344

# Begin BERT embeddings comparison
data = pd.read_pickle(f'manifesto/data/BertEmbeddingsTask/BertEmbeddingsTask__99914b932b-data.pkl')
data = pd.DataFrame(data)
data = pd.DataFrame(data['model'])
df1_normalized = pd.DataFrame(scaler.fit_transform(data['transcript_bert_embeddings'][0]))
df2_normalized = pd.DataFrame(scaler.fit_transform(data['pdf_bert_embeddings'][0]))
corr_df1 = df1_normalized.corr()
corr_df2 = df2_normalized.corr()


distances = distance.cdist(df1_normalized.values, df2_normalized.values, 'euclidean')

# fitting the MDS with n_components as 2
mds = MDS(n_components=2)
projected_distances = mds.fit_transform(distances)
plt.scatter(projected_distances[:, 0], projected_distances[:, 1])
plt.show()

# Assuming df1 and df2 are your dataframes, and that they are already normalized

# Let's assume df2 has more columns than df1
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
centers = kmeans.cluster_centers_
labels = kmeans.labels_

labels_series = pd.Series(labels, name="label")
# Concatenate the labels with the DataFrame
df1_labeled = pd.concat([reduced_df1, labels_series], axis=1)
df2_labeled = pd.concat([reduced_df2, labels_series], axis=1)
# Plot the results
plt.scatter(reduced_df1[:, 0], reduced_df1[:, 1], label='Video', alpha=0.5)
plt.scatter(reduced_df2[:, 0], reduced_df2[:, 1], label='Manifesto', alpha=0.5)
plt.scatter(centers[:, 0], centers[:, 1], s=200, alpha=0.5, c=df1_labeled["label"]) # Plot centers df1_labeled
plt.scatter(centers[:, 0], centers[:, 1], s=200, alpha=0.5, c=df2_labeled["label"]) # Plot centers df2_labeled
plt.legend()
plt.title('PCA Reduced Video & Manifesto BERT Embeddings')  # Set the title of the plot
plt.show()

# Reduce the dimensionality of each dataframe
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
dbscan = DBSCAN(eps=0.7, min_samples=5).fit(data)
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


# Create a pairplot
sns.pairplot(data)
plt.show()


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
df = data
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

