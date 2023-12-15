Most of the structure of the code for the experiments on the fMRI dataset is taken from this tutorial https://ja-che.github.io/hidimstat/auto_examples/plot_fmri_data_example.html#sphx-glr-auto-examples-plot-fmri-data-example-py, and adapted to the methods considered in our paper (IRKSN, IROSR, Lasso, ElasticNet, KSN, IRKSN, SRDI, OMP). To reproduce the figures and tables for the fMRI experiment, install a conda environment from the file `fmri_env.yml`, then run the `run_all.sh` script (or its version adapted to a cluster, `run_all_sbatch.sh`), for the two classes pair that are discussed in the paper, i.e. face/house and house/shoe, as such: 

```
bash ./run_all.sh face house [n_cores];
bash ./run_all.sh house shoe [n_cores]
```

(replace `[n_cores]` by the number of cores you wish to use for each of the python runs, for instance 16)

If you have access to a cluster, you can run instead:

```
sbatch ./run_all_sbatch.sh face house [n_cores]
sbatch ./run_all_sbatch.sh house shoe [n_cores]
```


Finally, run the `create_results.py` python script to generate the table of results.