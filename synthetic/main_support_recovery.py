import argparse
import numpy as np
from algorithms_support_recovery import get_score
import csv
import time
from benchopt.datasets.simulated import make_correlated_data
from sklearn.model_selection import train_test_split



def benchmark_model(algo, nruns, maxiter, n_samples, rho, snr, path):

    rs = [np.random.RandomState(i) for i in range(100)]
    n_features = 50
    mses = []
    for i in range(nruns):
        X, y, x0 = make_correlated_data(n_samples=n_samples, n_features=n_features, random_state=rs[i], rho=rho, snr=snr, density=0.2, X_density=1)
        mses.append(get_score(algo, rs[i], X, y, maxiter, x0))
    mse = np.mean(mses)
    std = np.std(mses)
    tstamp =  time.strftime("%D%H:%M", time.localtime(time.time()))
    data = [algo, mse, std, nruns, tstamp, X.shape[0], X.shape[1], n_samples, rho, snr]
    print(f'Score: mse {mse}, std:{std}')
    with open(path, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark sparse linear regression models')
    parser.add_argument('-a', '--algorithm', help='Algorithm to use')
    parser.add_argument('-n', '--nruns', type=int, help='Number of iterations')
    parser.add_argument('-m', '--maxiter', type=int, help='Max iter for iterative algos')
    parser.add_argument('-i', '--n_samples', type=int, default=30, help='Number of samples')
    parser.add_argument('-r', '--rho', type=float, default=0.5, help='Correlations')
    parser.add_argument('-s', '--snr', type=float, default=1., help='Signal to noise ratio')
    parser.add_argument('-p', '--path', type=str, help='Path for saving the results')
    args = parser.parse_args()

    benchmark_model(args.algorithm, args.nruns, args.maxiter, args.n_samples, args.rho, args.snr, args.path)

