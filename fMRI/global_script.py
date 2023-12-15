
import sys
from algorithms_and_support import get_score_and_beta
import numpy as np
import pandas as pd
from sklearn.utils import Bunch
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_extraction import image
from sklearn.linear_model import Ridge
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.image import mean_img
from nilearn.plotting import plot_stat_map, show

from hidimstat.stat_tools import zscore_from_pval, pval_from_scale
from hidimstat.standardized_svr import standardized_svr
from hidimstat.permutation_test import permutation_test, permutation_test_cv
from hidimstat.adaptive_permutation_threshold import ada_svr
from hidimstat.clustered_inference import clustered_inference
from hidimstat.ensemble_clustered_inference import ensemble_clustered_inference
from nilearn.plotting import plot_stat_map, show, view_img





def preprocess_haxby(subject=2, memory=None, class1='face', class2='house'):
    '''Gathering and preprocessing Haxby dataset for a given subject.'''

    # Gathering data
    haxby_dataset = datasets.fetch_haxby(subjects=[subject])
    fmri_filename = haxby_dataset.func[0]

    behavioral = pd.read_csv(haxby_dataset.session_target[0], sep=" ")

    # conditions = pd.DataFrame.to_numpy(behavioral['labels'])
    conditions = behavioral['labels'].values
    session_label = behavioral['chunks'].values

    condition_mask = np.logical_or(conditions == class1, conditions == class2)
    groups = session_label[condition_mask]

    # Loading anatomical image (back-ground image)
    if haxby_dataset.anat[0] is None:
        bg_img = None
    else:
        bg_img = mean_img(haxby_dataset.anat)

    # Building target where '1' corresponds to 'class1' and '-1' to 'class2'
    y = np.asarray((conditions[condition_mask] == class2) * 2 - 1)

    # Loading mask
    mask_img = haxby_dataset.mask
    masker = NiftiMasker(mask_img=mask_img, standardize=True,
                         smoothing_fwhm=None, memory=memory)

    # Computing masked data
    fmri_masked = masker.fit_transform(fmri_filename)
    X = np.asarray(fmri_masked)[condition_mask, :]

    return Bunch(X=X, y=y, groups=groups, bg_img=bg_img, masker=masker)



def plot_map(pval, one_minus_pval, zscore_threshold, title=None,
             cut_coords=[-25, -40, -5], masker=None, bg_img=None, output_file='default_output.png'):

    zscore = zscore_from_pval(pval, one_minus_pval)
    zscore_img = masker.inverse_transform(zscore)
    plot_stat_map(zscore_img, threshold=zscore_threshold, bg_img=bg_img,
                  dim=-1, cut_coords=cut_coords, title=title, output_file=output_file)
    
def train_method(algo):
    n_samples, n_features = X.shape
    beta_hat, final_score = get_score_and_beta(algo, np.random.RandomState(42), X, y,10000)
    std = np.linalg.norm(beta_hat) / np.sqrt(n_features)
    scale = std * np.ones(beta_hat.size)
    pval_std_svr, _, one_minus_pval_std_svr, _ = pval_from_scale(beta_hat, scale)
    print(final_score)

    target_fwer = 0.1
    # zscore_threshold_corr = zscore_from_pval((target_fwer / 2))
    correction = 1. / n_features
    zscore_threshold_no_clust = zscore_from_pval((target_fwer / 2) * correction)
    return beta_hat, final_score, pval_std_svr, one_minus_pval_std_svr, zscore_threshold_no_clust









if len(sys.argv) != 5:
    print("Usage: python script.py <algo> <class1> <class2> <n_threads>")
    sys.exit(1)

algo = sys.argv[1]
class1 = sys.argv[2]
class2 = sys.argv[3]
n_threads = sys.argv[4]

print("Algo:", algo)
print("Class1:", class1)
print("Class2:", class2)
print("n_threads:", n_threads)

import os
os.environ["OMP_NUM_THREADS"] = str(n_threads)




data = preprocess_haxby(subject=2, class1=class1, class2=class2)
X, y, groups, masker = data.X, data.y, data.groups, data.masker
mask = masker.mask_img_.get_fdata().astype(bool)
results = dict()
beta_hat, final_score, pval_std_svr, one_minus_pval_std_svr, zscore_threshold_no_clust = train_method(algo)
np.save('./'+algo+f'_beta_{class1}_{class2}.npy', beta_hat)
np.save('./'+algo+f'_final_score_{class1}_{class2}.npy', np.array(final_score))



beta_hat = np.load('./'+algo+f'_beta_{class1}_{class2}.npy')
final_score = np.load('./'+algo+f'_final_score_{class1}_{class2}.npy')
n_features = len(beta_hat)
std = np.linalg.norm(beta_hat) / np.sqrt(n_features)
scale = std * np.ones(beta_hat.size)
pval_std_svr, _, one_minus_pval_std_svr, _ = pval_from_scale(beta_hat, scale)
target_fwer = 0.1
zscore_threshold_corr = zscore_from_pval((target_fwer / 2))
correction = 1. / n_features
zscore_threshold_no_clust = zscore_from_pval((target_fwer / 2) * correction)
plot_map(pval_std_svr, one_minus_pval_std_svr, zscore_threshold_no_clust, masker=masker, bg_img=data.bg_img, output_file=f'./{algo}_{class1}_{class2}_fmri.png')