import networkx
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")

g = networkx.read_gml('karate.gml')

pos = networkx.spring_layout(g)
networkx.draw_networkx_nodes(g, pos, node_size=20)
networkx.draw_networkx_edges(g, pos, alpha=0.5)
plt.show()

import community

partition = community.best_partition(g)
len(partition)

induced_graph = community.induced_graph(partition, g)
induced_graph.nodes()

community.modularity(partition, g)

dendo = community.generate_dendrogram(g)


def draw_graph(graph, partition, draw_edges):
    size = float(len(set(partition.values())))
    pos = networkx.spring_layout(graph)
    count = 0.
    for com in set(partition.values()):
        count += 1.
        list_nodes = [nodes for nodes in partition.keys()
                      if partition[nodes] == com]
        networkx.draw_networkx_nodes(graph,
                                     pos, list_nodes, node_size=20,
                                     node_color=str(count / size))
    if draw_edges:
        networkx.draw_networkx_edges(graph, pos, alpha=0.5)
    plt.show()


draw_graph(g, dendo[0], True)

community.modularity(dendo[0], g)

induced_graph = community.induced_graph(dendo[0], g)
induced_graph.edges(data='weight')[0]

partition
