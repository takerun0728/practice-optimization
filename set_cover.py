from pyqubo import Array, Constraint
import neal
import math

U = {1, 2, 3, 4, 5, 6, 7, 8, 9}
V = [{1, 2, 3, 6, 7, 8, 9}, {1, 2, 5, 8}, {3, 4, 7}, {4, 5, 7}, {6, 9}]
N = len(U)
M = len(V)
LOGM =  int(math.ceil(math.log2(M)))

if __name__ == '__main__':
    sampler = neal.SimulatedAnnealingSampler()
    xm = Array.create('xm', M, vartype='BINARY')
    #xnm = Array.create('xnm', (N, M), vartype='BINARY')
    xnm = Array.create('xnm', (N, LOGM), vartype='BINARY')
    #Ha1 = Constraint(sum([(1 - sum([xnm[n, m] for m in range(M)]))**2 for n in range(N)]), label='Ha1')
    #Ha2 = Constraint(sum([(sum([(m + 1) * xnm[n, m] for m in range(M)]) - sum([xm[m] for m in range(M) if (n + 1) in V[m]]))**2 for n in range(N)]), label='Ha2')
    
    Ha = Constraint(sum([(1 + sum([2**m * xnm[n][m] for m in range(LOGM)])  - sum([xm[m] for m in range(M) if (n + 1) in V[m]]))**2 for n in range(N)]), label='Ha')
    Hb = sum([x for x in xm])
    #H = (Ha1 + Ha2) * 2 + Hb
    H = Ha * 2 + Hb
    model = H.compile()
    qubo, _ = model.to_qubo()
    samples = sampler.sample_qubo(qubo, num_sweeps=100, num_reads=100)
    decoded = model.decode_sample(samples.first.sample, vartype='BINARY')
    print(decoded.constraints())
    print(f'Energy:{decoded.energy}')

    sets = [m for m in range(M) if decoded.array('xm', m)]
    num = [(n + 1, 1 + sum([2**m * decoded.array('xnm', (n, m)) for m in range(LOGM)])) for n in range(N)]
    print(sets)
    print(num)