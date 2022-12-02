import numpy as np
import matplotlib.pyplot as plt
from pyqubo import Array, solve_ising

if __name__ == '__main__':
    data1 = np.random.normal(loc=[1, 1], scale=1.0, size=(30, 2))
    data2 = np.random.normal(loc=[-1, -1], scale=1.0, size=(30, 2))
    data = np.vstack([data1, data2])

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(data1[:,0], data1[:,1], 'bo')
    ax1.plot(data2[:,0], data2[:,1], 'ro')
    
    dists = np.zeros((len(data), len(data)))
    for i, y in enumerate(data):
        for j, x in enumerate(data[i+1:]):
            dists[i, i+j+1] = np.linalg.norm(x - y)

    spin = Array.create('spin', shape=len(data), vartype='SPIN')
    H = sum([dists[i, j] * spin[i] * spin[j] for i in range(len(data)) for j in range(len(data)) if dists[i, j] != 0])
    model = H.compile()
    h, j, _ = model.to_ising()
    solution = solve_ising(h, j)
    decoded = model.decode_sample(solution, vartype="SPIN")
    labels = np.array([decoded.array("spin", idx) for idx in range(len(data))])

    ax2.plot(data[labels==1, 0], data[labels==1, 1], 'bo')
    ax2.plot(data[labels==-1, 0], data[labels==-1, 1], 'ro')

    plt.show()

    pass