from pyqubo import Array, Constraint
import neal
import networkx as nx
import matplotlib.pyplot as plt
import math

NV = 7
NU = 4
NODES = range(NV)
EDGES = [(0, 1, 2), (1, 2, 2), (1, 3, 1), (0, 3, 2.2), (2, 3, 2.2), (3, 4, 2), (4, 5, 2.2), (4, 6, 2.2), (5, 6, 4)]
NE = len(EDGES)
U = [0, 2, 5, 6]
U_ = [v for v in NODES if not(v in U)]
MAX_LEV = math.floor(NV/2)

if __name__ == '__main__':
    g = nx.Graph()
    g.add_nodes_from(NODES)
    g.add_weighted_edges_from(EDGES)

    xvi = Array.create('xvi', (NV, MAX_LEV+1), vartype='BINARY')
    xei = Array.create('xei', (NE*2, MAX_LEV), vartype='BINARY')
    ye = Array.create('ye', NE, vartype='BINARY')
    yu_ = Array.create('yu_', NV - NU, vartype='BINARY') #not U

    Ha1 = Constraint((1 - sum([xvi[v, 0] for v in NODES]))**2, label='Ha1')
    Ha2 = Constraint(sum([(1 - sum([xvi[v, i] for i in range(MAX_LEV+1)]))**2 for v in U]), label='Ha2')
    Ha3 = Constraint(sum([(yu_[u_] - sum([xvi[v, i] for i in range(MAX_LEV+1)]))**2 for u_, v in enumerate(U_)]), label='Ha3')
    Ha4 = Constraint(sum([(ye[e] - sum([xei[e, i] + xei[e+NE, i] for i in range(MAX_LEV)]))**2 for e in range(NE)]), label='Ha4')
    Ha5 = Constraint(sum([sum([(xvi[v, i] - sum([xei[e, i-1] if v == edge[1] else xei[e+NE, i-1] for e, edge in enumerate(g.edges) if v in edge]))**2 for i in range(1, MAX_LEV+1)]) for v in NODES]), label='Ha5')
    Ha6 = Constraint(sum([sum([xei[e, i-1] * (2 - xvi[v2, i] - xvi[v1, i-1]) + xei[e+NE, i-1] * (2 - xvi[v1, i] - xvi[v2, i-1]) for i in range(1, MAX_LEV+1)]) for e, (v1, v2) in enumerate(g.edges)]), label='Ha6')
    Hb = sum([d['weight'] * ye[e] for e, (_, __, d) in enumerate(g.edges.data())])
    H = Ha1 + Ha2 + Ha3 + Ha4 + Ha5 + Ha6 + Hb * 0.1
    model = H.compile()
    qubo, _ = model.to_qubo()
    sampler = neal.SimulatedAnnealingSampler()
    samples = sampler.sample_qubo(qubo, num_reads=1000)
    decoded = model.decode_sample(samples.first.sample, vartype='BINARY')
    print(decoded.constraints())
    print(decoded.energy)

    v_colors = ['red' if i in U else 'cyan'for i in range(NV)]
    e_colors = ['red' if decoded.array('ye', e) else 'black' for e in range(NE)]
    labels = {v:f'{v}:{i}' for v in NODES for i in range(MAX_LEV+1) if decoded.array('xvi', (v, i))}
    for v in NODES:
        if not (v in labels): labels[v] = f'{v}'
    nx.draw(g, with_labels=True, node_color=v_colors, edge_color=e_colors, labels=labels)
    plt.show()
    pass
    

