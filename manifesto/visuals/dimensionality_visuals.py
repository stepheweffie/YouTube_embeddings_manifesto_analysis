import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster import hierarchy
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.cluster import KMeans


def cosine_similarity_plot(dataframe):
    df = dataframe
    # Assuming your dataframe is named "df" and the embeddings are in a column named "embeddings"
    # Calculate the cosine similarity between all pairs of embeddings in the dataframe
    similarity_matrix = cosine_similarity(df["embeddings"].tolist())
    # Plot the embeddings in a scatter plot based on their cosine similarity
    fig, ax = plt.subplots()
    scatter = ax.scatter(df["x"], df["y"], c=similarity_matrix.flatten())
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(scatter)
    plt.show()


def pca_kmeans_dimensionality(dataframe):
    # Apply PCA to reduce the dimensionality to 2D
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=5)
    labels = kmeans.fit_predict(dataframe)
    # Convert the labels to a Pandas Series object
    labels_series = pd.Series(labels, name="label")
    # Concatenate the labels with the DataFrame
    df_labeled = pd.concat([dataframe, labels_series], axis=1)
    pca = PCA(n_components=2)
    # pca_result = pca.fit_transform(df)
    pca_result = pca.fit_transform(df_labeled.drop(columns=["label"]))
    # Visualize the reduced-dimensional data using a scatter plot with color-coded labels
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df_labeled["label"])
    plt.show()


def hierarchy_plot(dataframe):
    df = dataframe
    # Assuming your dataframe is named "df" and the embeddings are in a column named "embeddings"
    # Calculate the cosine similarity between all pairs of embeddings in the dataframe
    similarity_matrix = cosine_similarity(df["embeddings"].tolist())
    # Perform hierarchical clustering on the similarity matrix
    linkage_matrix = hierarchy.linkage(similarity_matrix, method="complete")
    # Plot the resulting clustering tree
    fig, ax = plt.subplots()
    dendrogram = hierarchy.dendrogram(linkage_matrix, ax=ax)
    ax.set_xlabel("Embeddings")
    ax.set_ylabel("Distance")
    plt.show()


def tsne_reduce_dimensionality(pca_result):
    # Apply t-SNE to further reduce the dimensionality to 2D
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    tsne_result = tsne.fit_transform(pca_result)
    # Visualize the reduced-dimensional data using a scatter plot
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1])