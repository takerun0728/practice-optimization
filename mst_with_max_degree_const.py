from pyqubo import Array, LogEncInteger, Constraint
import neal
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np

DELTA = 3
N = 7
NODES = range(N)
MAX_LEV = math.floor(N / 2)
EDGES = [(0, 1, 1), (1, 2, 1), (1, 3, 1), (3, 5, 1), (4, 5, 1), (5, 6, 1), (0, 4, 2), (2, 6, 2), (2, 3, 1.4), (3, 4, 1.4)]
NE = len(EDGES)

if __name__ == '__main__':
    g = nx.Graph()
    g.add_nodes_from(NODES)
    g.add_weighted_edges_from(EDGES)
    gc = nx.complement(g)
    
    ye = Array.create('ye', NE, vartype='BINARY')
    xuvi = Array.create('xuv', (NE * 2, MAX_LEV), vartype='BINARY')
    xvi = Array.create('xvi', (len(NODES), MAX_LEV+1), vartype='BINARY')
    zv = [LogEncInteger(f'z{i}', (1, DELTA)) for i in range(N)]
    
    e_conv = {edge:e for e, edge in enumerate(g.edges)}
    Ha1 = Constraint((1 - sum([xvi[v, 0] for v in NODES]))**2, label='Ha1')
    Ha2 = Constraint(sum([(1 - sum([xvi[v, i] for i in range(MAX_LEV+1)]))**2 for v in range(N)]), label='Ha2')
    Ha3 = Constraint(sum((ye[e] - sum([xuvi[e, i-1] + xuvi[e+NE, i-1] for i in range(1,MAX_LEV+1)]))**2 for e in range(NE)), label='Ha3')
    Ha4 = Constraint(sum([sum([(xvi[v, i] - sum([xuvi[e, i-1] if v == edge[1] else xuvi[e+NE, i-1] for e, edge in enumerate(g.edges) if v in edge]))**2 for i in range(1, MAX_LEV+1)]) for v in NODES]), label='Ha4')
    Ha5 = Constraint(sum([sum([xuvi[e, i-1] * (2 - xvi[edge[0], i-1] - xvi[edge[1], i]) + xuvi[e+NE, i-1] * (2 - xvi[edge[1], i-1] - xvi[edge[0], i]) for i in range(1, MAX_LEV+1)]) for e, edge in enumerate(g.edges)]), 'Ha5')
    Ha6 = Constraint(sum([(zv[v] - sum([xuvi[e, i-1] + xuvi[e+NE, i-1] for e, edge in enumerate(g.edges) for i in range(1, MAX_LEV+1) if v in edge]))**2 for v in NODES]), label='Ha6')
    Hb = sum([ye[e] * d['weight'] for e, (_, __, d) in enumerate(g.edges.data())])

    H = Ha1 + Ha2 + Ha3 + Ha4 + Ha5 + Ha6 + Hb * 0.1
    model = H.compile()
    qubo, _ = model.to_qubo()
    sampler = neal.SimulatedAnnealingSampler()
    samples = sampler.sample_qubo(qubo, num_reads=3000)
    decoded = model.decode_sample(samples.first.sample, vartype='BINARY')
    print(decoded.constraints())
    print(decoded.energy)

    colors = ['red' if decoded.array('ye', e) else 'black' for e in range(NE)]
    labels = {v:f'{v}:{i}' for v in g.nodes for i in range(MAX_LEV+1) if decoded.array('xvi', (v, i))}
    nx.draw(g, with_labels=True, edge_color=colors, labels=labels)
    plt.show()
    pass
