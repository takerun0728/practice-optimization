from pyqubo import Array, Constraint
import neal
from itertools import combinations

U = {1, 2, 3, 4, 5, 6, 7, 8, 9}
V = [{1, 2, 3, 6, 9}, {1, 2, 5, 8}, {4, 7}, {4, 5}, {6, 9}]

if __name__ == '__main__':
    sampler = neal.SimulatedAnnealingSampler()
    xs = Array.create('xs', len(V), vartype='BINARY')
    Ha = Constraint(sum([0 if v1.isdisjoint(v2) else xs[i] * xs[j]  for (i, j), (v1, v2) in zip(combinations(range(len(V)), 2), combinations(V, 2))]), 'Ha')
    Hb = -sum([x for x in xs])
    H = Ha + Hb
    model = H.compile()
    qubo, _ = model.to_qubo()
    samples = sampler.sample_qubo(qubo)
    decoded = model.decode_sample(samples.first.sample, vartype='BINARY')
    print(decoded.constraints(only_broken=True))
    
    selected = [i for i in range(len(V)) if decoded.array('xs', i)]
    print(selected)
    check = set()
    for i in selected:
        check = check & V[i]
    print(check) 
    pass