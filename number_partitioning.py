import numpy as np
from pyqubo import Array, solve_ising

N = 10

if __name__ == '__main__':
    data = np.random.random(N)
    data /= data.sum()
    data = np.array([1, 2000, 1000, 1001])
    print(data)

    spin = Array.create('spin', shape=len(data), vartype='SPIN')
    H = (sum([data[i] * spin[i] for i in range(len(data))]))**2
    model = H.compile()
    h, j, _ = model.to_ising()
    solution = solve_ising(h, j)
    decoded = model.decode_sample(solution, vartype='SPIN')
    labels = np.array([decoded.array('spin', idx) for idx in range(len(data))])
    print(f"{data[labels>0]}: {data[labels>0].sum()})")
    print(f"{data[labels<0]}: {data[labels<0].sum()})")
