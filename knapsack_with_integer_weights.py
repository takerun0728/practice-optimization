from pyqubo import Array, Constraint
import neal
import math

C = [95, 60, 25, 12, 1]
W = [470, 300, 130, 70, 30]
N = len(C)
WMAX = 500
CMAX = max(C)
LOGW = int(math.ceil(math.log2(WMAX)))

if __name__ == '__main__':
    sampler = neal.SimulatedAnnealingSampler()
    xw = Array.create('xw', LOGW, vartype='BINARY')
    xi = Array.create('xi', N, vartype='BINARY')
    offset = WMAX - 2**LOGW + 1

    Ha = (offset + sum([2**w * xw[w] for w in range(LOGW)]) - sum([xi[i] * W[i] for i in range(N)]))**2
    Hb = -sum([xi[i] * C[i] for i in range(N)])
    H = Ha * 10 + Hb
    model = H.compile()
    qubo, _ = model.to_qubo()
    samples = sampler.sample_qubo(qubo, num_reads=100)
    decoded = model.decode_sample(samples.first.sample, vartype='BINARY')
    print(decoded.constraints())
    print(decoded.energy)

    selected = [i for i in range(N) if decoded.array('xi', i)]
    weight = offset + sum([2**w * decoded.array('xw', w) for w in range(LOGW)])
    print(selected)
    print(weight)