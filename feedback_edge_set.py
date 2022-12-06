from pyqubo import Array, Constraint
import neal
import networkx as nx
import matplotlib.pyplot as plt
import math

NV = 8
NODES = range(NV)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0), (7, 1), (7, 5), (4, 2)]
NE = len(EDGES)
MAX_LEV = math.floor(NV/2)

if __name__ == '__main__':
    g = nx.DiGraph()
    g.add_nodes_from(NODES)
    g.add_edges_from(EDGES)

    xvi = Array.create('xvi', (NV, MAX_LEV+1), vartype='BINARY')
    xei = Array.create('xei', (NE, MAX_LEV), vartype='BINARY')
    ye = Array.create('ye', NE, vartype='BINARY')
    Ha1 = Constraint(sum([(1 - sum(xvi[v]))**2 for v in NODES]), label='Ha1')
    Ha2 = Constraint(sum([(ye[e] - sum(xei[e]))**2 for e in range(NE)]), label='Ha2')
    Ha3 = Constraint(sum([xei[e, i-1] * (2 - xvi[v1, i-1] - sum([xvi[v2, j] for j in range(i, MAX_LEV+1)])) for e, (v1, v2) in enumerate(g.edges) for i in range(1, MAX_LEV+1)]), label='Ha3')
    
    Hb = -sum(ye)
    H = Ha1 + Ha2 + Ha3 * 0.2 + Hb * 0.02
    model = H.compile()
    qubo, _ = model.to_qubo()
    sampler = neal.SimulatedAnnealingSampler()
    samples = sampler.sample_qubo(qubo, num_reads=100000)
    decoded = model.decode_sample(samples.first.sample, vartype='BINARY')
    print(decoded.constraints())
    print(decoded.energy)
    
    colors = ['red' if decoded.array('ye', e) else 'grey' for e in range(NE)]
    labels = {v:f'{v}:{i}' for v in NODES for i in range(MAX_LEV+1) if decoded.array('xvi', (v, i))}
    for v in NODES:
        if not(v in labels): labels[v] = f'{v}'
    nx.draw(g, with_labels=True, edge_color=colors, labels=labels)
    plt.show()
    pass
    
