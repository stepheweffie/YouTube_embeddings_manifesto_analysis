import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster import hierarchy
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np


def cosine_similarity_plot(manifesto, transcript):
    X = manifesto
    Y = transcript

    # Calculate the cosine similarity between all pairs of embeddings in the dataframe
    similarity_matrix = cosine_similarity(X, Y, dense_output=True)

    # Assuming cosine_similarities is a 1D array of cosine similarities
    plt.figure(figsize=(8, 6))
    plt.hist(similarity_matrix, bins=50)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.show()


def pca_reduce_dimensionality(dataframe):
    # DataFrame of embeddings
    # Apply PCA to reduce the dimensionality to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(dataframe)
    plt.figure(figsize=(8,6))
    plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('2D PCA of Embeddings')
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
    similarity_matrix = cosine_similarity(df)
    # Perform hierarchical clustering on the similarity matrix
    linkage_matrix = hierarchy.linkage(similarity_matrix, method="complete")
    # Plot the resulting clustering tree
    fig, ax = plt.subplots()
    dendrogram = hierarchy.dendrogram(linkage_matrix, ax=ax)
    ax.set_xlabel("Embeddings")
    ax.set_ylabel("Distance")
    plt.show()


def heatmap(corr_df1, corr_df2):
    # Create a heatmap for each correlation matrix, and preserve the column labels
    sns.heatmap(corr_df1, ax=ax[0], cmap='coolwarm', cbar=False)
    ax[0].set_title('DataFrame 1 Correlation Heatmap')
    ax[0].set_xticklabels(ax[0].get_xmajorticklabels(), fontsize=10)

    sns.heatmap(corr_df2, ax=ax[1], cmap='coolwarm')
    ax[1].set_title('DataFrame 2 Correlation Heatmap')
    ax[1].set_xticklabels(ax[1].get_xmajorticklabels(), fontsize=10)

    plt.tight_layout()
    plt.show()


def tsne_reduce_dimensionality(pca_result):
    # Apply t-SNE to further reduce the dimensionality to 2D
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    tsne_result = tsne.fit_transform(pca_result)
    # Visualize the reduced-dimensional data using a scatter plot
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1])