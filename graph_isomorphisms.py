from pyqubo import Array, Constraint
import neal
import networkx as nx

NV = 5
NODES = range(NV)
EDGES1 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
EDGES2 = [(0, 2), (2, 4), (4, 1), (1, 3), (3, 0)]

if __name__ == '__main__':
    g1 = nx.Graph()
    g1.add_nodes_from(NODES)
    g1.add_edges_from(EDGES1)
    gc1 = nx.complement(g1)
    g2 = nx.Graph()
    g2.add_nodes_from(NODES)
    g2.add_edges_from(EDGES2)
    gc2 = nx.complement(g2)

    xvi = Array.create('xvi', (NV, NV), vartype='BINARY')
    Ha1 = Constraint(sum([(1 - sum([xvi[v, i] for v in range(NV)]))**2 for i in range(NV)]), label='Ha1')
    Ha2 = Constraint(sum([(1 - sum([xvi[v, i] for i in range(NV)]))**2 for v in range(NV)]), label='Ha2')
    Ha3 = Constraint(sum([xvi[v1, i1] * xvi[v2, i2] + xvi[v1, i2] * xvi[v2, i1] for v1, v2 in g1.edges for i1, i2 in gc2.edges]), label='Ha3')
    Ha4 = Constraint(sum([xvi[v1, i1] * xvi[v2, i2] + xvi[v1, i2] * xvi[v2, i1] for v1, v2 in gc1.edges for i1, i2 in g2.edges]), label='Ha4')
    H = Ha1 + Ha2 + Ha3 + Ha4
    model = H.compile()
    qubo, _ = model.to_qubo()
    sampler = neal.SimulatedAnnealingSampler()
    samples = sampler.sample_qubo(qubo)
    decoded = model.decode_sample(samples.first.sample, vartype='BINARY')
    print(decoded.constraints())
    print(decoded.energy)

    map = [(v, i) for v in range(NV) for i in range(NV) if decoded.array('xvi', (v, i))]
    print(map)
    remapped = []
    for e in EDGES1:
        for m in map:
            if e[0] == m[0]:
                a = m[1]
                break
        for m in map:
            if e[1] == m[0]:
                b = m[1]
                break
        remapped.append((a, b))
    print(remapped)