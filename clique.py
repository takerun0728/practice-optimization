import numpy as np
from pyqubo import Array, Constraint
import neal
import networkx as nx
import matplotlib.pyplot as plt

N = 5

if __name__ == '__main__':
    sampler = neal.SimulatedAnnealingSampler()

    while True:
        g = nx.fast_gnp_random_graph(N, 0.8)
        if nx.is_connected(g):
            break
    
    gc = nx.complement(g)
    cliques = Array.create('x', N, vartype='BINARY')

    Hb = Constraint(sum([cliques[edge_c[0]] * cliques[edge_c[1]] for edge_c in gc.edges()]), label='Hb')
    Hc = -sum([cliques[node] for node in g.nodes])
    H = Hb * 2 + Hc
    model = H.compile()
    qubo, offset = model.to_qubo()

    raw_solution = sampler.sample_qubo(qubo)
    decoded = model.decode_sample(raw_solution.first.sample, vartype='BINARY')

    colors = ['blue' if decoded.array('x', i) == 1 else 'red' for i in range(N)]
    print(decoded.constraints(only_broken=True))

    nx.draw(g, with_labels=True, node_color=colors)
    plt.show()