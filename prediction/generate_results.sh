#!/bin/bash

datasets=(leukemia housing breheny1 breheny2)
algorithms=(lasso enet irksn iht irosr ircr srdi omp ksn)

for dataset in "${datasets[@]}"
do
    for algo in "${algorithms[@]}"
    do
        echo "Running algorithm $algo on dataset $dataset"
        python main.py -d "$dataset" -a "$algo" -n 10 -m 500 || true
        # python main.py -d "$dataset" -a "$algo" -n 10 -m 1000 || true  # For BRHEE2006, need to run for 1000 iterations, and to update algorithms.py with the right hyperparams (see also README.md)
    done
done

echo "All commands have been executed"
