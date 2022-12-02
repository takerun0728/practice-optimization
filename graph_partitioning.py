import numpy as np
from pyqubo import Array, solve_ising
import networkx as nx
import matplotlib.pyplot as plt

N = 10

if __name__ == '__main__':
    while True:
        g = nx.fast_gnp_random_graph(N, 0.5)
        if nx.is_connected(g):
            break

    delta = max(dict(g.degree()).values())
    a = min(2 * delta, N) / 8

    spins = Array.create('spins', N, vartype='SPIN')
    Ha = (sum([spins[i] for i in range(N)])) ** 2
    Hb = (sum([(1 - spins[edge[0]] * spins[edge[1]]) / 2 for edge in g.edges()]))
    H = a * Ha + Hb
    model = H.compile()
    h, J, _ = model.to_ising()

    solution = solve_ising(h, J)
    decoded = model.decode_sample(solution, vartype='SPIN')
    colors = ['blue' if decoded.array('spins', i) == 1 else 'red' for i in range(N)]

    nx.draw(g, with_labels=True, node_color=colors)
    plt.show()
    
    