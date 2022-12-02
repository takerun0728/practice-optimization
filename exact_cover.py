from pyqubo import Array, Constraint
import neal

U = {1, 2, 3, 4, 5, 6, 7, 8, 9}
V = [{1, 2, 3, 6, 7, 8, 9}, {1, 2, 5, 8}, {3, 4, 7}, {4, 5}, {6, 9}]

if __name__ == '__main__':
    solver = neal.SimulatedAnnealingSampler()
    xs = Array.create('xs', len(V), vartype='BINARY')
    Ha = Constraint(sum([(1 - sum([x if i in V[j] else 0 for j, x in enumerate(xs)]))**2 for i in U]), 'Ha')
    Hb = -sum([x for x in xs])
    H = Ha * 10 + Hb
    model = H.compile()
    qubo, offset = model.to_qubo()
    solution = solver.sample_qubo(qubo)
    decoded = model.decode_sample(solution.first.sample, vartype='BINARY')
    print(decoded.constraints(only_broken=True))
    
    selected = [i for i in range(len(V)) if decoded.array('xs', i)]
    recovered = set()
    for i in selected:
        recovered = recovered | V[i]
    print(selected)
    print(recovered)