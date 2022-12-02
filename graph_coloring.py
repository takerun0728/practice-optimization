from pyqubo import Array, Constraint
import neal
import networkx as nx
import matplotlib.pyplot as plt

COLORS = ['blue', 'red', 'green', 'orangered', 'pink', 'grey', 'cyan', 'yellow']
NC = 3
NV = 6
NODES = range(0, NV)
EDGES = [(0, 1), (0, 2), (0, 4), (1, 4), (1, 3), (2, 4), (2, 5), (3, 4), (3, 5)]

if __name__ == '__main__':
    g = nx.Graph()
    g.add_nodes_from(NODES)
    g.add_edges_from(EDGES)
    
    sampler = neal.SimulatedAnnealingSampler()
    xvc = Array.create('xvc', (NV, NC), vartype='BINARY')
    H = Constraint(sum([(1 - sum([xvc[v, c] for c in range(NC)]))**2 for v in range(NV)]), label='1') + Constraint(sum([sum([xvc[e1, c] * xvc[e2, c] for c in range(NC)]) for e1, e2 in g.edges]), label='2')
    model = H.compile()
    qubo, _ = model.to_qubo()
    samples = sampler.sample_qubo(qubo)
    decoded = model.decode_sample(samples.first.sample, vartype='BINARY')
    print(decoded.constraints())

    colors = [COLORS[c] for v in range(NV) for c in range(NC) if decoded.array('xvc', (v, c))]
    nx.draw(g, with_labels=True, node_color=colors)
    plt.show()