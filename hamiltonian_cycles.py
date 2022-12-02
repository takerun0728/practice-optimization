from pyqubo import Array, Constraint
import neal
import networkx as nx
import matplotlib.pyplot as plt

N = 5
NODES = range(N)
EDGES = [(0, 1), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (3, 4), (4, 0)]

if __name__ == '__main__':
    g = nx.DiGraph()
    g.add_nodes_from(NODES)
    g.add_edges_from(EDGES)
    gc = nx.complement(g)
    
    sampler = neal.SimulatedAnnealingSampler()
    xni = Array.create('xni', (N, N), vartype='BINARY')
    Ha1 = Constraint(sum([(1 - sum([xni[n, i] for i in range(N)]))**2 for n in range(N)]), label='Ha1')
    Ha2 = Constraint(sum([(1 - sum([xni[n, i] for n in range(N)]))**2 for i in range(N)]), label='Ha2')
    Hb = Constraint(sum([sum([xni[n1, i] * xni[n2, i + 1] for i in range(N - 1)]) + xni[n1, N-1] * xni[n2, 0] for n1, n2 in gc.edges]), label='Hb')
    H = Ha1 + Ha2 + Hb
    model = H.compile()
    qubo, _ = model.to_qubo()
    samples = sampler.sample_qubo(qubo)
    decoded = model.decode_sample(samples.first.sample, vartype='BINARY')
    print(decoded.constraints())

    route = [n for i in range(N) for n in range(N) if decoded.array('xni', (n, i))]
    route_edge = [e for e in zip(route[:-1], route[1:])] + [(route[-1], route[0])]
    print(route)
    colors = ['red' if e in route_edge else 'black' for e in EDGES]
    nx.draw(g, with_labels=True, edge_color=colors)
    plt.show()