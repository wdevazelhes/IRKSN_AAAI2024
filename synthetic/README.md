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

For convenience we also provide our conda environment exported in the file `ksn_env.yml`.
To reproduce the experiments, run `generate_results_snr.sh`, `generate_results_n_samples.sh`, and `generate_results_rho.sh`, and `evolution_of_f1_score_with_iterations`.
