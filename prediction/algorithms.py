from sklearn.linear_model import lasso_path, enet_path, OrthogonalMatchingPursuit
from sklearn.model_selection import train_test_split
from modopt.opt.proximity import KSupportNorm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from scipy import sparse
import numpy as np
import itertools

def get_score(algorithm, random_state, X, y, maxiter): 

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.25)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=random_state, test_size=0.25)
    assert X_train.shape[0] >= 7 , X_train.shape[0]
    assert X_valid.shape[0] >= 7, X_valid.shape[0]
    assert X_test.shape[0] >= 7, X_test.shape[0]
    scl = make_pipeline(VarianceThreshold(), StandardScaler())
    X_train_c = scl.fit_transform(X_train)
    y_offset = np.mean(y_train)
    y_train_c = y_train - y_offset 
    X_train_c_mean = np.mean(X_train_c, axis=0)


    print('Training...')
    if algorithm == 'lasso':
        print('Fitting lasso path')
        _, coefs, _ = lasso_path(X_train_c, y_train_c, random_state=random_state)
        print('Lasso path fitted.')
        minmse = np.infty
        for j in range(coefs.shape[1]):
            beta = coefs[:, j]
            intercept = get_intercept(X_train_c_mean, y_offset, beta)
            currmse = get_mse(beta, X_valid, y_valid, scl, intercept)
            if currmse < minmse:
                minmse, minbeta = currmse + 0., beta + 0.

    elif algorithm == 'enet':
        lratio = [.1, .5, .7, .9, .95, .99, 1]
        finminmse = np.infty
        finminbeta = None
        for k in range(len(lratio)):
            _, coefs, _ = enet_path(X_train_c, y_train_c, l1_ratio=lratio[k], random_state=random_state)
            minmse = np.infty
            for j in range(coefs.shape[1]):
                beta = coefs[:, j]
                intercept = get_intercept(X_train_c_mean, y_offset, beta)
                currmse = get_mse(beta, X_valid, y_valid, scl, intercept)
                if currmse < minmse:
                    minmse, minbeta = currmse + 0., beta + 0.
            if minmse < finminmse:
                finminmse, finminbeta = minmse + 0., minbeta + 0.
        minbeta = finminbeta +0.

    if algorithm == 'omp':
        minmse = np.infty
        for k in np.linspace(1, min(X_train_c.shape[0], X_train_c.shape[1]), 5):
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=int(k), fit_intercept=False, normalize=False)
            omp.fit(X_train_c, y_train_c)
            beta = omp.coef_
            intercept = get_intercept(X_train_c_mean, y_offset, beta)
            currmse = get_mse(beta, X_valid, y_valid, scl, intercept)
            if currmse < minmse:
                minmse, minbeta = currmse, beta


    if algorithm == 'iht':
        finminmse = np.infty
        for (k, eta) in itertools.product(np.linspace(1, X_train_c.shape[1], 5), [0.0001, 0.001, 0.01, 0.1, 1]): 
            max_iter = maxiter
            beta = np.zeros(X_train_c.shape[1])
            print('starting to train IHT:')
            minmse = np.infty
            for it in range(max_iter):
                if sparse.issparse(X_train_c):
                    loss = np.mean((X_train_c * beta - y_train_c)**2)
                else:
                    loss = np.mean((X_train_c @ beta - y_train_c)**2)
                if sparse.issparse(X_train_c):
                    beta -= eta * 1/X_train_c.shape[0] * X_train_c.T * (X_train_c * beta - y_train_c)
                else:
                    beta -= eta * 1/X_train_c.shape[0] * X_train_c.T @ (X_train_c @ beta - y_train_c)
                beta = hard_threshold(beta, int(k))

                intercept = get_intercept(X_train_c_mean, y_offset, beta)
                currmse = get_mse(beta, X_valid, y_valid, scl, intercept)
 
                if currmse < minmse:
                    minmse, minbeta = currmse + 0., beta + 0.

                print(loss)
            print('IHT training finished')
            intercept = get_intercept(X_train_c_mean, y_offset, beta)
            currmse = get_mse(beta, X_valid, y_valid, scl, intercept)
            if minmse < finminmse:
                finminmse, finminbeta = minmse, minbeta + 0.
        minbeta = finminbeta + 0.


    if algorithm == 'ksn':
        minmse = np.infty
        for (lam, k, L) in itertools.product([0.1, 1.], np.linspace(1, X_train_c.shape[1], 5), [1000000]):
            max_iter = maxiter
            beta = np.zeros(X_train_c.shape[1])
            theta = 1.
            alpha = np.zeros(X_train_c.shape[1])
            prox = KSupportNorm(beta=lam/L, k_value=int(k))
            for it in range(max_iter):
                if sparse.issparse(X_train_c):
                    loss = np.mean((X_train_c * beta - y_train_c)**2) 
                else:
                    loss = np.mean((X_train_c @ beta - y_train_c)**2)
                print(loss)
                old_theta = theta + 0.
                theta = (1 + np.sqrt(1 + 4 * theta**2))/2
                old_beta = beta + 0.
                beta = prox.op(alpha - 1/L * X_train_c.T @ (X_train_c @ alpha - y_train_c))
                alpha = beta + (old_theta - 1)/theta * (beta - old_beta)
            print('KSN penalty training finished')
            intercept = get_intercept(X_train_c_mean, y_offset, beta)
            currmse = get_mse(beta, X_valid, y_valid, scl, intercept)
            if currmse < minmse:
                minmse, minbeta = currmse, beta + 0.

    if algorithm == 'irksn':
        minmse = np.infty
        for (alpha, k) in itertools.product([0.0001, 0.001, 0.01, 0.1, 1, 10], np.linspace(1, X_train_c.shape[1], 5)):
        # for (alpha, k) in itertools.product([0.6], [33]): # For BRHEE2006 (see README.md), (also need to run for 1000 iterations: update generate_results.sh with -m 1000)
            max_iter = maxiter
            beta = np.zeros(X_train_c.shape[1])
            beta_sum = beta + 0.
            beta_avg = beta + 0.
            v = np.zeros(X_train_c.shape[0])
            z = np.zeros(X_train_c.shape[0])
            z_old = np.zeros(X_train_c.shape[0])
            nuclear_norm = np.linalg.norm(X_train_c, ord='nuc')
            gamma = alpha * nuclear_norm**(-2)
            theta = 1
            prox = KSupportNorm(beta=(1 - alpha)/alpha , k_value=int(k))
            min_val_loss = np.infty
            curr_loss = np.infty
            for it in range(max_iter):
                if it % 5 == 0:
                    intercept = get_intercept(X_train_c_mean, y_offset, beta)
                    curr_val_loss = get_mse(beta, X_valid, y_valid, scl, intercept)
                    if curr_val_loss < min_val_loss:
                        min_val_loss, min_beta_avg = curr_val_loss, beta
                z_old = z + 0.
                if sparse.issparse(X_train_c):
                    beta = prox.op(- 1/alpha * X_train_c.T * z)
                    r = prox.op(- 1/alpha * X_train_c.T * v)
                    z = v + gamma * (X_train_c * r - y_train_c)
                else:
                    beta = prox.op(- 1/alpha * X_train_c.T @ z)
                    r = prox.op(- 1/alpha * X_train_c.T @ v)
                    z = v + gamma * (X_train_c @ r - y_train_c)
                theta_old = theta + 0.
                theta = (1 + np.sqrt(1 + 4 * theta ** 2))/2
                v = z + (theta_old - 1)/(theta) * (z - z_old)
            print('IRKSN training finished')
            intercept = get_intercept(X_train_c_mean, y_offset, min_beta_avg)
            currmse = get_mse(min_beta_avg, X_valid, y_valid, scl, intercept)
            if currmse < minmse:
                minmse, minbeta = currmse, min_beta_avg + 0.
                print(alpha, k)

    if algorithm == 'ircr':
        minmse = np.infty
        max_iter = maxiter
        beta = np.zeros(X_train_c.shape[1])
        beta_sum = beta + 0.
        beta_avg = beta + 0.
        theta = np.zeros(X_train_c.shape[0])
        theta_old = theta + 0.
        nuclear_norm = np.linalg.norm(X_train_c, ord='nuc')
        tau = 0.9 * 1/np.sqrt(2 * nuclear_norm**2)
        sigma = 0.9 * 1/np.sqrt(2 * nuclear_norm**2)
        min_val_loss = np.infty
        curr_loss = np.infty

        for it in range(max_iter):
            if it % 5 == 0:
                intercept = get_intercept(X_train_c_mean, y_offset, beta_avg)
                curr_val_loss = get_mse(beta_avg, X_valid, y_valid, scl, intercept)
                if curr_val_loss < min_val_loss:
                    min_val_loss, min_beta_avg = curr_val_loss, beta_avg
            if sparse.issparse(X_train_c):
                curr_loss = np.mean((X_train_c * beta - y_train_c)**2) 
            else:
                curr_loss = np.mean((X_train_c @ beta - y_train_c)**2)
            print(curr_loss)
            beta = shrink(beta - tau * X_train_c.T @ (2 * theta - theta_old), tau)
            theta_old = theta + 0.
            theta = theta_old + sigma * (X_train_c @ beta - y_train_c)
            beta_sum = beta_sum + beta
            beta_avg = beta_sum / it
        print('IRCR penalty training finished')
        minbeta = min_beta_avg +0.
        print('Done.')

    if algorithm == 'irosr':
        minmse = np.infty
        for (eta, alpha) in itertools.product([0.0001, 0.001, 0.01, 0.1, 1], [0.0001, 0.001, 0.01, 0.1, 1]):
            alpha = 0.001
            max_iter = maxiter
            min_val_loss = np.infty
            curr_loss = np.infty
            u, v = alpha * np.ones(X_train_c.shape[1]), alpha * np.zeros(X_train_c.shape[1])
            beta = u**2 - v** 2
            for it in range(max_iter):
                if it % 5 == 0:
                    intercept = get_intercept(X_train_c_mean, y_offset, beta)
                    curr_val_loss = get_mse(beta, X_valid, y_valid, scl, intercept)
                    if curr_val_loss < min_val_loss:
                        min_val_loss, minbeta_tmp = curr_val_loss, beta
                curr_loss = firosr(X_train_c, u, v, y_train_c) 
                print(curr_loss)
                if sparse.issparse(X_train_c):
                    return NotImplementedError
                else:
                    u_grad, v_grad = gradirosr(X_train_c, u, v, y_train_c)
                    u -= eta * u_grad
                    v -= eta * v_grad
                    beta = u**2 - v** 2 
            print('IROSR penalty training finished')
            intercept = get_intercept(X_train_c_mean, y_offset, minbeta_tmp)
            currmse = get_mse(minbeta_tmp, X_valid, y_valid, scl, intercept)
            if currmse < minmse:
                minmse, minbeta = currmse, minbeta_tmp + 0.

    if algorithm == 'srdi':
        minmse = np.infty
        for (kappa, alpha) in itertools.product([0.0001, 0.001, 0.01, 0.1, 1], [0.0001, 0.001, 0.01, 0.1, 1]):
            max_iter = maxiter
            min_val_loss = np.infty
            curr_loss = np.infty
            z = np.zeros(X_train_c.shape[1])
            beta = np.zeros(X_train_c.shape[1])
            for it in range(max_iter):
                if it % 5 == 0:
                    intercept = get_intercept(X_train_c_mean, y_offset, beta)
                    curr_val_loss = get_mse(beta, X_valid, y_valid, scl, intercept)
                    if curr_val_loss < min_val_loss:
                        min_val_loss, minbeta_tmp = curr_val_loss, beta
                curr_loss = np.mean((X_train_c @ beta - y_train_c)**2) 
                print(curr_loss)
                if sparse.issparse(X_train_c):
                    return NotImplementedError
                else:
                    z = z + alpha/X.shape[1] * X_train_c.T @ (y_train_c - X_train_c @ beta)
                    beta = kappa * shrink(z, 1)
            print('srdi penalty training finished')
            intercept = get_intercept(X_train_c_mean, y_offset, minbeta_tmp)
            currmse = get_mse(minbeta_tmp, X_valid, y_valid, scl, intercept)
            if currmse < minmse:
                minmse, minbeta = currmse, minbeta_tmp + 0.

    print('Training done. Evaluating')
    final_mse = get_mse(minbeta, X_test, y_test, scl, y_offset)
    print('Done.')
    return final_mse

def get_mse(beta, X, y, scl, intercept):
    X_scaled = scl.transform(X)
    mse = np.mean((X_scaled @ beta + intercept - y)**2)
    return mse

def get_intercept(X_train_c_mean, y_offset, beta):
    return y_offset - X_train_c_mean @ beta

def hard_threshold(arr, k):
    top_k_indices = np.argpartition(np.abs(arr), -k)[-k:]
    thresholded_arr = np.zeros_like(arr)
    thresholded_arr[top_k_indices] = arr[top_k_indices]
    return thresholded_arr

def shrink(u, tau, factor=1.):
    """Soft-thresholding of vector u at level tau > 0."""
    return np.sign(u) * np.maximum(0., np.abs(u) - tau)

def firosr(X, u, v, y):
    return 1/(2 * X.shape[0]) * np.sum((X @ (u * u - v * v) - y)**2)

def gradirosr(X, u, v, y):
    residual = (X @ (u * u - v * v) - y)
    return (2/X.shape[0] * X.T * u[:, None] @ (residual), -2/X.shape[0] * X.T * v[:, None] @ (residual))