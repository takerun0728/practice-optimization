from pyqubo import Array, Constraint
import neal
import numpy as np
import networkx as nx
from itertools import combinations, product
import matplotlib.pyplot as plt

EX = np.array([[1, 0, -1, 1, 0, 1], [0, -1, -1, 0, 0, 0], [-1, 1, 1, 0, 0, 0]])
N = EX.shape[1]

if __name__ == '__main__':
    g = nx.Graph()
    g.add_nodes_from([(i, j) for i, c in enumerate(EX) for j, n in enumerate(c) if n])
    g.add_edges_from([(n1, n2) for n1, n2 in combinations(g.nodes, 2) if n1[0] == n2[0]])

    pos_list = [[j for j, n in enumerate(EX[:, i]) if n == 1] for i in range(N)]
    neg_list = [[j for j, n in enumerate(EX[:, i]) if n == -1] for i in range(N)]
    
    tmp = [((i, j1), (i, j2)) for i in range(N) for j1, j2 in product(pos_list[i], neg_list[i])]
    g.add_edges_from([((j1, i), (j2, i)) for i in range(N) for j1, j2 in product(pos_list[i], neg_list[i])])

    sampler = neal.SimulatedAnnealingSampler()
    conv = {n:i for i, n in enumerate(g.nodes)}
    xs = Array.create('xs', len(g.nodes), vartype='BINARY')
    Ha = Constraint(sum([xs[conv[i]] * xs[conv[j]] for i, j in g.edges]), label='Ha')
    Hb = -sum([x for x in xs])
    H = Ha * 2 + Hb
    model = H.compile()
    qubo, _ = model.to_qubo()
    samples = sampler.sample_qubo(qubo)
    decoded = model.decode_sample(samples.first.sample, vartype='BINARY')
    print(decoded.constraints(only_broken=True))
    
    colors = ['red' if decoded.array('xs', i) else 'blue' for i in range(len(g.nodes))]

    nx.draw(g, with_labels=True, node_color=colors)
    plt.show()

    pass