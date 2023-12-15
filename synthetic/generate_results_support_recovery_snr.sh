#!/bin/bash

algorithms=(lasso enet irksn iht irosr ircr srdi omp ksn)

for snr in 0.1 0.5 1. 2. 3.
do
    for algo in "${algorithms[@]}"
    do
        echo "Running algorithm $algo for support recovery XP"
        python main_support_recovery.py -a "$algo" -n 5 -m 20000 -s $snr --path 'results_snr.csv'|| true
    done
done

echo "All commands have been executed"
