from pyqubo import Array, LogEncInteger, Constraint
import neal
import math

L = [7, 5, 3, 2, 2, 2, 2]
NJ = len(L)
NC = 3
LMAX = max(L)
LOG_LMAX = math.ceil(math.log2(LMAX))

if __name__ == '__main__':
    sampler = neal.SimulatedAnnealingSampler()
    xjc = Array.create('xjc', (NJ, NC), vartype='BINARY')
    ylc = [LogEncInteger(f'yl{c+1}', (0, LMAX)) for c in range(NC-1)]

    c0_total = sum([L[j] * xjc[j, 0] for j in range(NJ)])
    Ha1 = Constraint(sum([(1 - sum([xjc[j, c] for c in range(NC)]))**2 for j in range(NJ)]), label='Ha1')
    Ha2 = Constraint(sum([(ylc[c-1] - c0_total + sum([L[j] * xjc[j, c] for j in range(NJ)]))**2 for c in range(1, NC)]), label='Ha2')
    Hb = c0_total   
    H = LMAX * (Ha1 + Ha2) + Hb
    model = H.compile()
    qubo, _ = model.to_qubo()
    samples = sampler.sample_qubo(qubo, num_reads=1000)
    decoded = model.decode_sample(samples.first.sample, vartype='BINARY')
    print(decoded.constraints())
    print(decoded.energy)

    for c in range(NC):
        job_list = [L[j] for j in range(NJ) if decoded.array('xjc', (j, c))]
        if c == 0:
            print(f"{job_list}, {sum(job_list)}")
        else:
            print(f"{job_list}, {sum(job_list)}, {decoded.subh[f'yl{c}']}")
    pass


    