import os
import sys
import time
import numpy as np
from numpy import linalg as LA
from joblib import Parallel, delayed
from math import sqrt
from sklearn.base import BaseEstimator, TransformerMixin

import rec_new

floatX = np.float64
EPS = 1e-8
class TranSIV(BaseEstimator, TransformerMixin):
    def __init__(self, n_components_1=100, n_components_2=100,n_components_3=100,max_iter=10,batch_size=1000, n_jobs=4,
                 init_std=0.01, random_state=None, save_params=False,
                 save_dir='.', early_stopping=False, verbose=False, **kwargs):

        self.n_components_1 = n_components_1
        self.n_components_2 =n_components_2
        self.n_components_3 =n_components_3
        self.max_iter = max_iter
        self.init_std = init_std
        self.random_state = random_state
        self.save_params = save_params
        self.save_dir = save_dir
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        self._parse_kwargs(**kwargs)
    def _parse_kwargs(self, **kwargs):

        self.lam_gamma = float(kwargs.get('lambda_gamma', 1e-5))
        self.lam_theta = float(kwargs.get('lambda_theta', 1e-5))
        self.lam_beta = float(kwargs.get('lambda_beta', 1e-5))
        self.lam_y = float(kwargs.get('lam_y', 1.0))
        self.lam_s = float(kwargs.get('lam_s', 1.0))
        self.init_mu = float(kwargs.get('init_mu', 0.01))
        self.init_yita = float(kwargs.get('init_yita', 0.01))
        self.a1 = float(kwargs.get('a1', 1.0))
        self.b1 = float(kwargs.get('b1', 4.0))
        self.a2 = float(kwargs.get('a2', 1.0))
        self.b2 = float(kwargs.get('b2', 4.0))
    def _init_params(self, n_users, n_items):

        self.gamma_c = self.init_std * \
                     np.random.randn(n_users, self.n_components_1).astype(floatX)
        self.gamma_s =self.init_std * \
                     np.random.randn(n_users, self.n_components_2).astype(floatX)

        self.theta_c = self.init_std * \
                     np.random.randn(n_users, self.n_components_1).astype(floatX)
        self.theta_s = self.init_std * \
                       np.random.randn(n_users, self.n_components_2).astype(floatX)
        self.theta_r = self.init_std * \
                       np.random.randn(n_users, self.n_components_3).astype(floatX)
        self.beta_c = self.init_std * \
                    np.random.randn(n_items, self.n_components_1).astype(floatX)
        self.beta_r = self.init_std * \
                      np.random.randn(n_items, self.n_components_3).astype(floatX)

        self.mu = self.init_mu * np.ones(n_items, dtype=floatX)
        self.yita = self.init_yita * np.ones(n_users, dtype=floatX)
    def fit(self, X, S, vad_data=None, **kwargs):
        n_users, n_items = X.shape
        self._init_params(n_users, n_items)
        self._update(X, S, vad_data, **kwargs)
        return self
    def _update(self, X, S, vad_data, **kwargs):
        n_users = X.shape[0]
        XT = X.T.tocsr()  # pre-compute this
        ST = S.T.tocsr()
        self.vad_ndcg = -np.inf
        for i in xrange(self.max_iter):
            if self.verbose:
                print('ITERATION #%d' % i)
            self._update_factors(X, S, XT)
            self._update_expo(X, S, n_users)
            if vad_data is not None:
                vad_ndcg = self._validate(X, vad_data, **kwargs)
                if self.early_stopping and self.vad_ndcg > vad_ndcg:
                    l = 0
                    # break  # we will not save the parameter for this iteration
                self.vad_ndcg = vad_ndcg

            if self.save_params:
                self._save_params(i)
        pass
    def _update_factors(self, X, S, XT):
        if self.verbose:
            start_t = _writeline_and_time('\tUpdating theta_c factors...')
        self.theta_c = recompute_theta_c(self.gamma_c, self.beta_c, self.theta_c,self.gamma_s,self.beta_r,self.theta_r,self.theta_s, X, S,
                                     self.lam_theta / self.lam_y,
                                     self.lam_y,
                                     self.lam_s,
                                     self.mu,
                                     self.yita,
                                     self.n_jobs,
                                     batch_size=self.batch_size
                                     )
        if self.verbose:
            print('\r\tUpdating theta_c factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating theta_r factors...')
        self.theta_r = recompute_factors(self.theta_c,self.beta_c,self.theta_r,self.beta_r,X,
                                         self.lam_theta / self.lam_y,
                                         self.lam_y,
                                         self.mu,
                                         self.n_jobs,
                                         batch_size=self.batch_size
                                         )
        if self.verbose:
            print('\r\tUpdating theta_r factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating theta_s factors...')
        self.theta_s = recompute_factors(self.theta_c, self.gamma_c,self.theta_s,self.gamma_s, S,
                                       self.lam_gamma / self.lam_s,
                                       self.lam_s,
                                       self.yita,
                                       self.n_jobs,
                                       batch_size=self.batch_size
                                       )
        if self.verbose:
            print('\r\tUpdating theta_s factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating beta_c factors...')
        self.beta_c = recompute_factors(self.beta_r,self.theta_r,self.beta_c,self.theta_c, XT,
                                       self.lam_theta / self.lam_y,
                                       self.lam_y,
                                       self.mu,
                                       self.n_jobs,
                                       batch_size=self.batch_size
                                       )
        if self.verbose:
            print('\r\tUpdating beta_c factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating beta_r factors...')
        self.beta_r = recompute_factors(self.beta_c,self.theta_c,self.beta_r,self.theta_r, XT,
                                       self.lam_theta / self.lam_y,
                                       self.lam_y,
                                       self.mu,
                                       self.n_jobs,
                                       batch_size=self.batch_size
                                       )
        if self.verbose:
            print('\r\tUpdating beta_r factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating gamma_c factors...')
        self.gamma_c = recompute_factors(self.gamma_s,self.theta_s,self.gamma_c,self.theta_c, S,
                                         self.lam_gamma / self.lam_s,
                                         self.lam_s,
                                         self.yita,
                                       self.n_jobs,
                                       batch_size=self.batch_size
                                       )
        if self.verbose:
            print('\r\tUpdating gamma_c factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating gamma_s factors...')
        self.gamma_s = recompute_factors(self.gamma_c,self.theta_c,self.gamma_s,self.theta_s, S,
                                         self.lam_gamma / self.lam_s,
                                         self.lam_s,
                                         self.yita,
                                       self.n_jobs,
                                       batch_size=self.batch_size
                                       )
        if self.verbose:
            print('\r\tUpdating gamma_s factors: time=%.2f'
                  % (time.time() - start_t))
            sys.stdout.flush()
        pass
    def _update_expo(self, X, S, n_users):
        if self.verbose:
            start_t = _writeline_and_time('\tUpdating visibility prior...')
        start_idx = range(0, n_users, self.batch_size)
        end_idx = start_idx[1:] + [n_users]
        A_sum = np.zeros_like(self.mu)

        AA = Parallel(n_jobs=self.n_jobs)(delayed(A_row_batch)(X[lo:hi], self.theta_c[lo:hi], self.beta_c,self.theta_r[lo:hi],self.beta_r,
                                                               self.lam_y, self.mu)
                                          for lo, hi in zip(start_idx, end_idx))
        for i in AA:
            A_sum += i
        self.mu = (self.a1 + A_sum - 1) / (self.a1 + self.b1 + n_users - 2)
        B_sum = np.zeros_like(self.yita)
        BB = Parallel(n_jobs=self.n_jobs)(delayed(A_row_batch)(S[lo:hi], self.theta_c[lo:hi], self.gamma_c,self.theta_s[lo:hi],self.gamma_s,
                                                               self.lam_s, self.yita)
                                          for lo, hi in zip(start_idx, end_idx))
        for i in BB:
            B_sum += i
        self.yita = (self.a2 + B_sum - 1) / (self.a2 + self.b2 + n_users - 2)

        if self.verbose:
            print('\r\tUpdating visibility prior: time=%.2f'
                  % (time.time() - start_t))
            sys.stdout.flush()
        pass

    def _validate(self, X, vad_data, **kwargs):
        '''Compute validation metric (NDCG@k)'''

        vad_rec_10 = rec_new.recall_at_k(X, vad_data,
                                          self.theta_c,
                                          self.beta_c,self.theta_r,self.beta_r, mu=self.mu,
                                          k=10)

        if self.verbose:
            print('\tValidation rec@10: %.4f' % vad_rec_10)
            sys.stdout.flush()
        return vad_rec_10

    def _save_params(self, iter):
        '''Save the parameters'''
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        filename = 'mod.npz'
        np.savez(os.path.join(self.save_dir, filename), U1=self.theta_c,
                 V1=self.beta_c,U2=self.theta_r,V2=self.beta_r, mu=self.mu)


def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()

def get_row(Y, i):
    '''Given a scipy.sparse.csr_matrix Y, get the values and indices of the
    non-zero values in i_th row'''
    lo, hi = Y.indptr[i], Y.indptr[i + 1]
    return Y.data[lo:hi], Y.indices[lo:hi]


def a_row_batch(Y_batch, theta_c, beta_c,theta_s,beta_s, lam_y, mu):
    '''Compute the posterior of visibility latent variables A by batch'''
    pEX = sqrt(lam_y / (2 * np.pi)) * \
          np.exp(-lam_y * (theta_c.dot(beta_c.T)+theta_s.dot(beta_s.T)) ** 2 / 2)
    A = (pEX + EPS) / (pEX + EPS + (1 - mu) / mu)
    A[Y_batch.nonzero()] = 1.
    return A

def A_row_batch(Y_batch, theta_c, beta_c,theta_s,beta_s, lam_y, mu):
    '''Compute the posterior of visibility latent variables A by batch'''
    pEX = sqrt(lam_y / (2 * np.pi)) * \
          np.exp(-lam_y * (theta_c.dot(beta_c.T)+theta_s.dot(beta_s.T)) ** 2 / 2)
    A = (pEX + EPS) / (pEX + EPS + (1 - mu) / mu)
    A[Y_batch.nonzero()] = 1.
    return A.sum(axis=0)

def _solve_theta_c(k, A_k, B_k,beta_c, gamma_c,beta_r,gamma_s,theta_s,theta_r, Y, S, f, lam, lam_y, lam_s, mu, yita):
    s_u, i_u = get_row(Y, k)
    s2_u, i2_u = get_row(S, k)
    a = np.dot(lam_y*s_u * (A_k[i_u]-theta_r.dot(beta_r[i_u].T)), beta_c[i_u]) + np.dot(lam_s*s2_u * (B_k[i2_u]-theta_s.dot(gamma_s[i2_u].T)), gamma_c[i2_u])
    B = lam_y*beta_c.T.dot(A_k[:, np.newaxis] * beta_c) + lam * np.eye(f) + lam_s*gamma_c.T.dot(B_k[:, np.newaxis] * gamma_c)
    return LA.solve(B, a)

def _solve_batch_theta_c(lo, hi, gamma_c, beta_c,theta_c,gamma_s,beta_r,theta_r,theta_s, Y, S, m, f, lam, lam_y, lam_s, mu, yita):
    assert theta_c.shape[0] == hi - lo
    A_batch = a_row_batch(Y[lo:hi], theta_c, beta_c, theta_r,beta_r,lam_y, mu)
    B_batch = a_row_batch(S[lo:hi], theta_c, gamma_c,theta_s,gamma_s,lam_s, yita)
    X_batch = np.empty_like(theta_c, dtype=theta_c.dtype)
    for ib, k in enumerate(xrange(lo, hi)):
        X_batch[ib] = _solve_theta_c(k, A_batch[ib], B_batch[ib], beta_c, gamma_c,beta_r,gamma_s,theta_s[ib],theta_r[ib], Y, S, f, lam, lam_y, lam_s, mu, yita)
    return X_batch
def recompute_theta_c(gamma_c, beta_c, theta_c,gamma_s,beta_r,theta_r,theta_s,
                      Y, S, lam, lam_y, lam_s, mu, yita,n_jobs, batch_size=1000):
    m, n = Y.shape
    assert beta_c.shape[0] == n
    assert theta_c.shape[0] == m
    f = beta_c.shape[1]
    start_idx = range(0, m, batch_size)
    end_idx = start_idx[1:] + [m]
    res = Parallel(n_jobs=n_jobs)(delayed(_solve_batch_theta_c)(
        lo, hi, gamma_c, beta_c, theta_c[lo:hi], gamma_s,beta_r,theta_r[lo:hi],theta_s[lo:hi],
        Y, S, m, f, lam, lam_y, lam_s, mu, yita)
           for lo, hi in zip(start_idx, end_idx))
    X_new = np.vstack(res)
    return X_new

def _solve(k, A_k, beta_r,theta_c,beta_c, Y, f, lam, lam_y, mu):
    '''Update one single factor'''
    s_u, i_u = get_row(Y, k)
    a = np.dot(lam_y*s_u * (A_k[i_u]-theta_c.dot(beta_c[i_u].T)), beta_r[i_u])
    B = lam_y*beta_r.T.dot(A_k[:, np.newaxis] * beta_r) + lam * np.eye(f)
    return LA.solve(B, a)

def _solve_batch(lo, hi,beta_c,theta_c,beta_r,theta_r, Y, m, f, lam, lam_y, mu):
    '''Update factors by batch, will eventually call _solve() on each factor to
    keep the parallel process busy'''
    assert theta_c.shape[0] == hi - lo

    if mu.size == beta_c.shape[0]:  # update users
        A_batch = a_row_batch(Y[lo:hi], theta_c, beta_c,theta_r,beta_r, lam_y, mu)
    else:  # update items
        A_batch = a_row_batch(Y[lo:hi], theta_c, beta_c,theta_r,beta_r, lam_y, mu[lo:hi,
                                                               np.newaxis])
    X_batch = np.empty_like(theta_r, dtype=theta_r.dtype)
    for ib, k in enumerate(xrange(lo, hi)):
        X_batch[ib] = _solve(k, A_batch[ib], beta_r,theta_c[ib],beta_c, Y, f, lam, lam_y, mu)
    return X_batch

def recompute_factors(theta_c,beta_c,theta_r,beta_r, Y, lam, lam_y, mu,n_jobs, batch_size=1000):
    '''Regress X to Y with visibility matrix (computed on-the-fly with X_old) and
    ridge term lam by embarrassingly parallelization. All the comments below
    are in the view of computing user factors'''
    m, n = Y.shape  # m = number of users, n = number of items
    assert beta_r.shape[0] == n
    assert theta_r.shape[0] == m
    f = beta_r.shape[1]  # f = number of factors
    start_idx = range(0, m, batch_size)
    end_idx = start_idx[1:] + [m]
    res = Parallel(n_jobs=n_jobs)(delayed(_solve_batch)(
        lo, hi, beta_c, theta_c[lo:hi],beta_r,theta_r[lo:hi], Y, m, f, lam, lam_y, mu)
           for lo, hi in zip(start_idx, end_idx))
    X_new = np.vstack(res)
    return X_new