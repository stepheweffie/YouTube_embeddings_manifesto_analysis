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
    G = nx.Graph()
    # Add edges to the graph for each correlation
    for column in corr_matrx.columns:
        for index in corr_matrx.index:
            # We only add an edge between different nodes and if the correlation is above a threshold
            if index != column and corr_matrx.loc[index, column] > 0.7:
                G.add_edge(index, column, weight=corr_matrx.loc[index, column])

    # Draw the resulting graph
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos)

    plt.show()
