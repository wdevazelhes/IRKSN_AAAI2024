#!/bin/bash

algorithms=(lasso enet irksn iht irosr ircr srdi omp ksn)

for n_samples in 10 30 50 70 90
do
    for algo in "${algorithms[@]}"
    do
        echo "Running algorithm $algo for support recovery XP"
        python main_support_recovery.py -a "$algo" -n 5 -m 20000 --n_samples $n_samples --path 'results_n_samples.csv' || true
    done
done

echo "All commands have been executed"
