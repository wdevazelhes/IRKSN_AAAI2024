#!/bin/bash

algorithms=(lasso enet irksn iht irosr ircr srdi omp ksn)

for rho in 0.1 0.3 0.5 0.7 0.9
do
    for algo in "${algorithms[@]}"
    do
        echo "Running algorithm $algo for support recovery XP"
        python main_support_recovery.py -a "$algo" -n 5 -m 20000 --rho $rho --path 'results_rho.csv' || true
    done
done

echo "All commands have been executed"
