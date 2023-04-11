# main python file that test some more stuff
import numpy as np


# import networkx
import networkx as nx

# create a graph that represents the grid
# G = nx.grid_2d_graph(5,5)
# nx.draw(G)

# show the graph that in a rectangular grid

from matplotlib import pyplot as plt

plt.figure(figsize=(6,6))

x_size = 20
y_size = 20

G = nx.grid_2d_graph(x_size,y_size)

# Set all weights to 1
for edge in G.edges:
    G.edges[edge]['weight'] = 1

pos = {(x,y):(y,-x) for x,y in G.nodes()}

G.add_edges_from([
    ((x, y), (x+1, y+1))
    for x in range(x_size-1)
    for y in range(y_size-1)
] + [
    ((x+1, y), (x, y+1))
    for x in range(x_size-1)
    for y in range(y_size-1)
], weight=1.4)

nx.draw(G, pos=pos,
        node_color='grey',
        with_labels=False,
        node_size=10)
print('')