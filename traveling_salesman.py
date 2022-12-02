from pyqubo import Array, Constraint
import neal
import networkx as nx
import matplotlib.pyplot as plt

N = 5
NODES = range(N)
EDGES = [(0, 1, 10), (0, 4, 5), (1, 2, 5), (1, 3, 5), (2, 3, 5), (2, 4, 10), (3, 4, 5)]
EDGES.extend([(e[1], e[0], e[2]) for e in EDGES])

if __name__ == '__main__':
    g = nx.DiGraph()
    g.add_nodes_from(NODES)
    g.add_weighted_edges_from(EDGES)
    gc = nx.complement(g)

    xni = Array.create('xni', (N, N), vartype='BINARY')
    Ha1 = Constraint(sum([(1 - sum([xni[n, i] for i in range(N)]))**2 for n in range(N)]), label='Ha1')
    Ha2 = Constraint(sum([(1 - sum([xni[n, i] for n in range(N)]))**2 for i in range(N)]), label='Ha2')
    Ha3 = Constraint(sum([sum([xni[n1, i] * xni[n2, i+1] for i in range(N-1)]) + xni[n1, N-1] * xni[n2, 0] for n1, n2 in gc.edges]), label='Ha3')
    Hb = sum([sum([d['weight'] * xni[n1, i] * xni[n2, i+1] for i in range(N-1)]) + d['weight'] * xni[n1, N-1] * xni[n2, 0] for n1, n2, d in g.edges.data()])
    max_w = max([e[2] for e in EDGES])
    H = (Ha1 + Ha2 + Ha3) * max_w * 2 + Hb
    model = H.compile()
    qubo, _ = model.to_qubo()
    sampler = neal.SimulatedAnnealingSampler()
    samples = sampler.sample_qubo(qubo)
    decoded = model.decode_sample(samples.first.sample, vartype='BINARY')
    print(decoded.constraints())
    print(decoded.energy)

    route = [n for i in range(N) for n in range(N) if decoded.array('xni', (n, i))]
    route_edges = [(e1, e2) for e1, e2 in zip(route[:-1], route[1:])] + [(route[-1], route[0])]
    colors = ['red' if e in route_edges else 'black' for e in g.edges]
    width = [3.0 if c == 'red' else 1.0 for c in colors]
    nx.draw(g, with_labels=True, width=width, edge_color=colors)
    plt.show()