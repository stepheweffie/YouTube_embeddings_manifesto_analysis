import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# Assume we have a correlation matrix as a pandas DataFrame
corr_matrix = pd.DataFrame(
    np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.1],
        [0.3, 0.1, 1.0]
    ]),
    columns=['A', 'B', 'C'],
    index=['A', 'B', 'C']
)


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

    for column in corr_matrx:
        for index in corr_matrx:
            # We only add an edge between different nodes and if the correlation is above a threshold
            # if index != column and corr_matrx.loc[index, column]:
            G.add_edge(index, column, weight=corr_matrx.loc[index, column])

    # Draw the resulting graph
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos)

    plt.show()
