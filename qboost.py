from tkinter import Place
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import datasets, metrics
from pyqubo import Array, Placeholder, Constraint, solve_qubo
import neal

NUM_CLF = 32
SAMPLE_TRAIN = 40
NUM_TRAIN = 450

def calc_accuracy(models, weights, x, y):
    y_pred_list = []
    for model in models:
        y_pred_list.append(model.predict(x))
    
    y_pred_list = np.array(y_pred_list)
    y_pred = np.sign(np.sum(y_pred_list * weights.reshape(-1, 1), axis=0))
    return metrics.accuracy_score(y_true=y, y_pred=y_pred)

if __name__ == '__main__':
    cancer_data = datasets.load_breast_cancer()
    data_noisy = np.concatenate([cancer_data.data, np.random.rand(cancer_data.data.shape[0], 100)], axis=1)
    print(data_noisy.shape)

    labels = (cancer_data.target - 0.5) * 2

    x_train = data_noisy[:NUM_TRAIN, :]
    x_test = data_noisy[NUM_TRAIN:, :]
    y_train = labels[:NUM_TRAIN]
    y_test = labels[NUM_TRAIN:]

    models = [DTC(splitter='random', max_depth=1) for i in range(NUM_CLF)]
    for model in models:
        train_idx = np.random.choice(np.arange(x_train.shape[0]), SAMPLE_TRAIN)
        model.fit(X=x_train[train_idx], y=y_train[train_idx])

    y_pred_list_train = []
    for model in models:
        y_pred_list_train.append(model.predict(x_train))
    y_pred_list_train = np.array(y_pred_list_train)
    y_pred_train = np.sign(y_pred_list_train)
    
    print(calc_accuracy(models, np.ones(NUM_CLF), x_test, y_test))

    weights = Array.create('weight', shape=NUM_CLF, vartype='BINARY')
    lam = Placeholder('lam')
    H_clf = sum([(sum(w * y_p for w, y_p in zip(weights, y_ps)) / NUM_CLF - y)**2 for y_ps, y in zip(y_pred_train.T, y_train)])
    H_norm = Constraint(sum([w for w in weights]), 'norm')
    H = H_clf + H_norm * lam
    model = H.compile()
    qubo, _ = model.to_qubo(feed_dict={'lam':10})
    sampler = neal.SimulatedAnnealingSampler()
    solution = sampler.sample_qubo(qubo, num_reads=10000)
    weights = [solution.first.sample[f'weight[{i}]'] for i in range(NUM_CLF)]
    print(calc_accuracy(models, np.array(weights), x_test, y_test))

    pass



