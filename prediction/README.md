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
To reproduce the experiments, run `generate_results.sh`. Results can be printed in a more convenient way with `read_results.ipynb`.

(Note: for IRKSN's results on BRHEE2006, there are 2 lines of code that need to be uncommented, one in `algorithms.py`, and one in `generate_results.sh`, see the corresponding file for more info, and the Appendix of the paper for more details).
