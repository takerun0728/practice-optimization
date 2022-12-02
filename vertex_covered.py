from pyqubo import Array, Constraint
import neal
import networkx as nx
import matplotlib.pyplot as plt

N = 10

if __name__ == '__main__':
    sampler = neal.SimulatedAnnealingSampler()
    xs = Array.create('xs', N, vartype='BINARY')
    
    while True:
        g = nx.fast_gnp_random_graph(N, 0.2)
        if nx.is_connected(g):
            break

    Ha = Constraint(sum([(1 - xs[v1]) * (1 - xs[v2]) for v1, v2 in g.edges]), 'Ha')
    Hb = sum([x for x in xs])
    H = Ha + Hb
    model = H.compile()
    qubo, _ = model.to_qubo()
    samples = sampler.sample_qubo(qubo)
    decoded = model.decode_sample(samples.first.sample, vartype='BINARY')
    print(decoded.constraints(only_broken=True))

    colors = ['red' if decoded.array('xs', i) else 'blue' for i in range(N)]
    nx.draw(g, with_labels=True, node_color=colors)
    plt.show()
