import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

edges = [
    ("S1","S2",0.1),
    ("S1","S3",0.2),
    ("S1","S4",0.5),
    ("S2","S1",0.2),
    ("S2","S3",0.4),
    ("S2","S4",0.1),
    ("S3","S2",0.5),
    ("S3","S4",0.1),
    ("S4","S1",0.3),
    ("S4","S2",0.1),
    ("S4","S3",0.2)
]

for u,v,w in edges:
    G.add_edge(u,v,weight=w)

pos = nx.spring_layout(G)

nx.draw(G, pos, with_labels=True, node_size=2000)
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels={(u,v):d['weight'] for u,v,d in G.edges(data=True)}
)

plt.show()