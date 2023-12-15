import argparse
import numpy as np
from datasets import get_data
from algorithms import get_score
import csv
import time

def benchmark_model(dataset, algo, nruns, maxiter):
    X, y = get_data(dataset)
    random_state = np.random.RandomState(42)
    mses = []
    for i in range(nruns):
        mses.append(get_score(algo, random_state, X, y, maxiter))
    mse = np.mean(mses)
    std = np.std(mses)
    tstamp =  time.strftime("%D%H:%M", time.localtime(time.time()))
    data = [dataset, algo, mse, std, nruns, tstamp, X.shape[0], X.shape[1]]
    print(f'Score: mse {mse}, std:{std}')
    with open('results.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark sparse linear regression models')
    parser.add_argument('-d', '--dataset', help='Type of dataset to use')
    parser.add_argument('-a', '--algorithm', help='Algorithm to use')
    parser.add_argument('-n', '--nruns', type=int, help='Number of iterations')
    parser.add_argument('-m', '--maxiter', type=int, help='Max iter for iterative algos')
    args = parser.parse_args()

    benchmark_model(args.dataset, args.algorithm, args.nruns, args.maxiter)

