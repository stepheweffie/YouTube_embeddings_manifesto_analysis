import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pandas as pd


def tsne_kmeans(dataframe):
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(dataframe)
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=0)  # choose appropriate number of clusters
    kmeans.fit(tsne_results)
    labels = kmeans.predict(tsne_results)
    # Create a scatter plot of the t-SNE output, colored by K-means cluster assignment
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1],c=labels, cmap='viridis', alpha=0.6)
    plt.show()

