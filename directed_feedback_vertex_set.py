from pyqubo import Array, Constraint
import neal
import networkx as nx
import matplotlib.pyplot as plt

NV = 8
NODES = range(NV)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0), (7, 1), (7, 5), (4, 2)]

if __name__ == '__main__':
    g = nx.DiGraph()
    g.add_nodes_from(NODES)
    g.add_edges_from(EDGES)

    yv = Array.create('yv', NV, vartype='BINARY')
    xvi = Array.create('xvi', (NV, NV), vartype='BINARY')
    
    Ha1 = Constraint(sum([(yv[v] - sum([xvi[v, i] for i in NODES]))**2 for v in range(NV)]), label='Ha1')
    Ha2 = Constraint(sum([sum([xvi[v1, i] * xvi[v2, j] for i in range(NV) for j in range(i, NV)]) for v1, v2 in g.edges]), label='Ha2')
    Hb = -sum([yv[v] for v in NODES])
    H = Ha1 + Ha2 + Hb * 0.5
    model = H.compile()
    qubo, _ = model.to_qubo()
    sampler = neal.SimulatedAnnealingSampler()
    samples = sampler.sample_qubo(qubo)
    decoded = model.decode_sample(samples.first.sample, vartype='BINARY')
    print(decoded.constraints())
    print(decoded.energy)

    colors = ['cyan' if decoded.array('yv', v) else 'grey' for v in NODES]

    nx.draw(g, with_labels=True, node_color=colors)
    plt.show()

