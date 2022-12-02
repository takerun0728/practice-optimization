from pyqubo import Array, Constraint
import networkx as nx
import neal
import matplotlib.pyplot as plt

COLORS = ['blue', 'red', 'green', 'orangered', 'pink', 'grey', 'cyan', 'yellow']
NC = 2
NV = 6
NODES = range(0, NV)
EDGES = [(0, 1), (0, 2), (0, 4), (1, 4), (1, 3), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5)]

if __name__ == '__main__':
    g = nx.Graph()
    g.add_nodes_from(NODES)
    g.add_edges_from(EDGES)

    sampler = neal.SimulatedAnnealingSampler()
    xvc = Array.create('xvc', (NV, NC), vartype='BINARY')
    Ha = Constraint(sum([(1 - sum([xvc[v, c] for c in range(NC)]))**2 for v in range(NV)]), label='Ha')
    each_v_num = [sum([xvc[v, c] for v in range(NV)]) for c in range(NC)]
    max_edge_num = [each_v_num[c] * (each_v_num[c] - 1) / 2 for c in range(NC)]
    Hb = Constraint(sum([max_edge_num[c] - sum([xvc[e1, c] * xvc[e2, c] for e1, e2 in g.edges]) for c in range(NC)]), label='Hb')
    H = Ha + Hb
    model = H.compile()
    qubo, _ = model.to_qubo()
    samples = sampler.sample_qubo(qubo)
    decoded = model.decode_sample(samples.first.sample, vartype='BINARY')
    print(decoded.constraints())

    colors = [COLORS[c] for v in range(NV) for c in range(NC) if decoded.array('xvc', (v, c))]
    nx.draw(g, with_labels=True, node_color=colors)
    plt.show()    