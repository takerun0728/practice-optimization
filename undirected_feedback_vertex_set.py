from pyqubo import Array, Constraint
import neal
import networkx as nx
import matplotlib.pyplot as plt
import math

NV = 8
NODES = range(NV)
EDGES = [(0, 1), (2, 3), (3, 4), (4, 5), (6, 7), (7, 0), (1, 7), (2, 4)]
NE = len(EDGES)
MAX_LEV = math.floor(NV / 2)

if __name__ == '__main__':
    g = nx.Graph()
    g.add_nodes_from(NODES)
    g.add_edges_from(EDGES)

    yv = Array.create('yv', NV, vartype='BINARY')
    ye = Array.create('ye', NE*2, vartype='BINARY')
    xvi = Array.create('xvi', (NV, MAX_LEV+1), vartype='BINARY')
    xei = Array.create('xei', (NE*2, MAX_LEV), vartype='BINARY')
    Ha1 = Constraint(sum([(1 - yv[v] - sum([xvi[v, i] for i in range(MAX_LEV+1)]))**2 for v in NODES]), label='Ha1')
    Ha2 = Constraint(sum([(1 - ye[e] - ye[e+NE] - sum([xei[e, i] + xei[e+NE, i] for i in range(MAX_LEV)]))**2 for e in range(NE)]), label='Ha2')
    Ha3 = Constraint(sum([(ye[e] - yv[v2])**2 + (ye[e+NE] - yv[v1])**2 for e, (v1, v2) in enumerate(g.edges)]), label='Ha3')
    Ha4 = Constraint(sum([(xvi[v, i] - sum([xei[e, i-1] if v == v2 else xei[e+NE, i-1] for e, (v1, v2) in enumerate(g.edges) if v in (v1, v2)]))**2 for i in range(1, MAX_LEV+1) for v in NODES]), label='Ha4')
    Ha5 = Constraint(sum([xei[e, i-1] * (2 - xvi[v1, i-1] - xvi[v2, i]) + xei[e+NE, i-1] * (2 - xvi[v2, i-1] - xvi[v1, i]) for i in range(1, MAX_LEV+1) for e, (v1, v2) in enumerate(g.edges)]), label='Ha5')
    Hb = sum(yv) * 0.1
    H = Ha1 + Ha2 + Ha3 + Ha4 + Ha5 + Hb
    model = H.compile()
    qubo, _ = model.to_qubo()
    sampler = neal.SimulatedAnnealingSampler()
    samples = sampler.sample_qubo(qubo, num_reads=1000)
    decoded = model.decode_sample(samples.first.sample, vartype='BINARY')
    print(decoded.constraints())
    print(decoded.energy)

    node_colors = ['red' if decoded.array('yv', v) else 'cyan' for v in NODES]
    edge_colors = ['red' if decoded.array('ye', e) or decoded.array('ye', e+NE) else 'black' for e in range(NE)]
    labels = {v:f'{v}:{i}' for v in NODES for i in range(MAX_LEV+1) if decoded.array('xvi', (v, i))}
    for v in NODES:
        if not (v in labels): labels[v] = f'{v}'
    nx.draw(g, with_labels=True, node_color=node_colors, edge_color=edge_colors, labels=labels)
    plt.show()
    pass