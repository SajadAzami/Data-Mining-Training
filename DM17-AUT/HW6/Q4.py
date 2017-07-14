import networkx
import matplotlib.pyplot as plt
import community

g = networkx.read_gml('karate.gml')

s = networkx.spring_layout(g)
networkx.draw_networkx_nodes(g, s, node_size=50)
networkx.draw_networkx_edges(g, s, alpha=1)
plt.show()

partition = community.best_partition(g)
induced_graph = community.induced_graph(partition, g)
print community.modularity(partition, g)
dg = community.generate_dendrogram(g)


def draw_graph(graph, partition, draw_edges):
    size = float(len(set(partition.values())))
    s = networkx.spring_layout(graph)
    count = 0
    colors = ['b',
              'g',
              'r',
              'c',
              'm',
              'y',
              'k']
    for com in set(partition.values()):
        count += 1
        nodes_list = [nodes for nodes in partition.keys()
                      if partition[nodes] == com]
        print str(count / size)
        networkx.draw_networkx_nodes(graph,
                                     s, nodes_list, node_size=50,
                                     node_color=colors[count])
    if draw_edges:
        networkx.draw_networkx_edges(graph, s, alpha=1)
    plt.show()


draw_graph(g, dg[0], True)

print partition
