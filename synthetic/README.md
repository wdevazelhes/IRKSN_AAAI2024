Necessary packages: 

- numpy
- scipy
- scikit-learn
- modopt
- benchopt
- rpy2
- tqdm
- matplotlib
- pandas
- libsvmdata
- appdirs

They can be installed from scratch with the following commands:

```bash
$ conda create -n ksn -c conda-forge numpy scipy scikit-learn tqdm matplotlib pandas rpy2
$ conda activate ksn
$ pip install libsvmdata benchopt jupyter modopt appdirs
```

For convenience, we also provide our conda environment exported in the file `ksn_env.yml`.
To reproduce from scratch the experiments, first, delete the content of the results files, i.e. delete the content of `results_n_samples.csv`, `results_rho.csv` and `results_snr.csv` (but keep the `.csv` file themselves). Then, run `generate_results_support_recovery_snr.sh`, `generate_results_support_recovery_n_samples.sh`, and `generate_results_support_recovery_rho.sh`. To plot the results once they are generated, run the notebook `read_results.ipynb`. Finally, to plot the figure describing the evolution of the f1-score along iterations, run the notebook `evolution_of_f1_score_with_iterations`.
