import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Assume we have a correlation matrix as a pandas DataFrame


def networkx_graph(corr_matrx):
    # Create a networkx graph
    # Add edges to the graph for each correlation
    # Create a mask to ignore self-correlation and correlations below a threshold
    mask = np.where((corr_matrx == 1) | (corr_matrx < 0.69), 0, corr_matrx)

    # Create graph from the masked correlation matrix
    G = nx.to_networkx_graph(mask)

    # Draw the graph
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True)
    plt.show()

