from pyqubo import Array, Constraint
import neal
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

N = 6
NODES = range(N)
EDGES = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (2, 5)]

if __name__ == '__main__':
    g = nx.Graph()
    g.add_nodes_from(NODES)
    g.add_edges_from(EDGES)

    max_dim = max(list(dict(g.degree).values()))
    a = (max_dim - 2) * 2

    conv = {edge:i for i, edge in enumerate(g.edges)}
    conv.update({(edge[1], edge[0]):i for i, edge in enumerate(g.edges)})

    sampler = neal.SimulatedAnnealingSampler()
    xs = Array.create('xs', len(EDGES), vartype='BINARY')
    Ha = Constraint(sum([xs[conv[(n, e1)]] * xs[conv[(n, e2)]] for n in g.nodes for e1, e2 in combinations(g[n], 2)]), label='Ha')
    lam = [sum([xs[conv[(n, e)]] for e in g[n]]) for n in g.nodes]
    Hb = Constraint(sum([(1 - lam[e1]) * (1 - lam[e2]) for e1, e2 in g.edges]), label='Hb')
    Hc = sum([x for x in xs])
    H = a * Ha + Hb + 0.5 * Hc
    model = H.compile()
    qubo, _ = model.to_qubo()
    samples = sampler.sample_qubo(qubo)
    decoded = model.decode_sample(samples.first.sample, vartype='BINARY')
    print(decoded.constraints(only_broken=True))

    colors = ['red' if decoded.array('xs', i) else 'black' for i in range(len(g.edges))]
    nx.draw(g, with_labels=True, edge_color=colors)
    plt.show()