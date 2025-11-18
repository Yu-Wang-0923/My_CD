# -*- coding: utf-8 -*-
"""
功能聚类模型 - FunClu 类
基于参考代码实现
"""
import torch
import numpy as np
from sklearn.cluster import KMeans
import scipy
import scipy.optimize
import pandas as pd
import random
from scipy.linalg import toeplitz
import os

# 设置 PyTorch 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'  # 强制使用 CPU
torch.set_default_device(device)
torch.set_default_dtype(torch.float64)


def get_SAD1(pars, d):
    """SAD1 covariance structure.
    return a dxd size of covariance matrix regardless of times(length=d)
    only accept two parameters to model such matrix"""
    phi, gamma = pars[0], pars[1]
    diag = (1 - phi ** (2 * np.linspace(1, d, d))) / (1 - phi ** 2)
    SIGMA = diag * phi ** scipy.linalg.toeplitz(range(0, d), range(0, d))
    SIGMA = gamma ** 2 * (np.tril(SIGMA) + np.triu(SIGMA.T) - np.diag(np.full(d, diag)))
    return SIGMA


def get_mean(par, d):
    mu = par[0] * d ** par[1]
    return mu


def get_mean_init(x, y, mean_func, n_mean_pars, maxit=1e5):
    """n_mean_pars: length of parameters control mean"""
    params_0 = np.ones(n_mean_pars)
    par_est = scipy.optimize.curve_fit(mean_func, xdata=x, ydata=y, p0=params_0, maxfev=int(maxit))[0]
    return par_est


def linear_equation(x, *pars):
    """Linear model to fit mean."""
    a, b = pars[0], pars[1]
    y = a * x + b
    return y


def power_equation(x, *pars):
    """power equation to model mean."""
    a, b = pars[0], pars[1]
    y = a * x ** b
    return y


def np_fourier_series(x, *pars):
    nn = (len(pars) - 2) / 2
    a0, a, b, w = pars[0], pars[1:int(nn + 1)], pars[int(nn + 1):int(2 * nn + 1)], pars[-1]
    series = 0.0
    for deg in range(int(nn)):
        series += a[deg] * np.cos((deg + 1) * w * x) + b[deg] * np.sin((deg + 1) * w * x)
    return series + a0


def np_logistic_growth(x, *pars):
    """logistic growth equation to model mean."""
    a, b, r = pars[0], pars[1], pars[2]
    y = a / (1 + b * np.exp(-r * x))
    return y


def fourier_series(x, *pars):
    nn = (len(pars) - 2) / 2
    a0, a, b, w = pars[0], pars[1:int(nn + 1)], pars[int(nn + 1):int(2 * nn + 1)], pars[-1]
    series = 0.0
    for deg in range(int(nn)):
        series += a[deg] * torch.cos((deg + 1) * w * x) + b[deg] * torch.sin((deg + 1) * w * x)
    return series + a0


def get_SAD1_tied(pars, d):
    """SAD1 covariance structure.
    return a dxd size of covariance matrix regardless of times(length=d)
    only accept two parameters to model such matrix"""
    phi, gamma = pars[0], pars[1]
    diag = (1 - phi ** (2 * torch.tensor(range(1, d + 1), device=device, dtype=torch.float64))) / (1 - phi ** 2)
    SIGMA = diag * phi ** torch.tensor(scipy.linalg.toeplitz(range(0, d), range(0, d)), device=device, dtype=torch.float64)
    SIGMA = gamma ** 2 * (SIGMA.tril() + SIGMA.T.triu() - torch.diag(diag))
    return SIGMA


def get_AR1_tied(pars, d):
    """AR1 covariance structure.
     return a dxd size of covariance matrix regardless of times(length=d)
     only accept two parameters to model such matrix"""
    sigma, phi = pars[0], pars[1]
    base = torch.tensor(toeplitz(torch.linspace(1, d, d)), device=device, dtype=torch.float64)
    V = 1 / (1 - phi ** 2) * phi ** base
    SIGMA = sigma ** 2 * V
    return SIGMA


def get_SAD1_full(pars, d):
    """SAD1 covariance structure.
    return k(k = number of cluster) dxd size of covariance matrix regardless of times(length=d)
    only accept k*2 parameters"""
    SIGMAs = torch.stack(list(map(lambda k: get_SAD1_tied(k, d=d), pars)))
    return SIGMAs


def get_AR1_full(pars, d):
    """AR1 covariance structure.
     return k(k = number of cluster) dxd size of covariance matrix regardless of times(length=d)
     only accept k*2 parameters"""
    SIGMAs = torch.stack(list(map(lambda k: get_AR1_tied(k, d=d), pars)))
    return SIGMAs


def get_full():
    pass


def get_tied():
    pass


def get_diag():
    pass


def get_spherical():
    pass


def get_SAD1_inv(pars, d):
    phi, gamma = pars[0], pars[1]
    diag_element = (1.0 + phi ** 2) / gamma ** 2
    diag_element1 = - phi / gamma ** 2
    sigma_inv = torch.eye(d, device=device, dtype=torch.float64)
    sigma_inv.diagonal().copy_(diag_element)
    sigma_inv.diagonal(1).copy_(diag_element1)
    sigma_inv.diagonal(-1).copy_(diag_element1)
    return sigma_inv


def get_SAD1_det_log(pars, d):
    phi, gamma = pars[0], pars[1]
    sigma_det_log = 2 * torch.tensor(d, device=device, dtype=torch.float64) * torch.log(gamma)
    return sigma_det_log


def get_SAD1_L(pars, d):
    """cholesky decomposition of SAD1"""
    phi, gamma = pars[0], pars[1]
    sigma = gamma * phi ** torch.tensor(scipy.linalg.toeplitz(range(0, d)), device=device, dtype=torch.float64)
    sigma_L = sigma.tril()
    return sigma_L


def _n_cov_parameters(n_components, n_features, covariance_type):
    """Return the number of free cov parameters in the model."""
    if covariance_type == "full":
        cov_params = n_components * n_features * (n_features + 1) / 2.0
    elif covariance_type == "diag":
        cov_params = n_components * n_features
    elif covariance_type == "tied":
        cov_params = n_features * (n_features + 1) / 2.0
    elif covariance_type == "spherical":
        cov_params = n_components
    return int(cov_params)


MODEL_Collections = {
    'name_mean': ['None', 'power_equation', 'logistic_growth', 'linear_equation', "fourier_series1", "fourier_series2"],
    'functions_mean': [None, power_equation, np_logistic_growth, linear_equation, fourier_series, fourier_series],
    'mean_closed_form': [True, False, False, False, False, False],
    'functions_mean_init': [None, power_equation, np_logistic_growth, linear_equation, np_fourier_series, np_fourier_series],
    'n_pars_mean': [None, 2, 3, 2, 4, 6],
    'name_cov': ['SAD1_tied', 'AR1_tied', 'full', 'tied', 'diag', 'spherical', 'SAD1_full', 'AR1_full'],
    'functions_cov': [get_SAD1_tied, get_AR1_tied, get_full, get_tied, get_diag, get_spherical, get_SAD1_full, get_AR1_full],
    'cov_closed_form': [False, False, True, True, True, True, False, False],
    'n_pars_cov': [2, 2, None, None, None, None, None, None]
}


def _estimate_log_gaussian_prob(X, means, covariances):
    K = means.shape[0]
    if (len(covariances.shape) != 3):
        covariances = covariances.repeat(K, 1, 1)
    mvn_log = list(
        map(lambda x, y: torch.distributions.MultivariateNormal(x, y).log_prob(X), means, covariances))
    return torch.stack(mvn_log)


def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    """Estimate the full covariance matrices."""
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[:: n_features + 1] += reg_covar
    return covariances


def _estimate_gaussian_covariances_tied(resp, X, nk, means, reg_covar):
    """Estimate the tied covariance matrix."""
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot(nk * means.T, means)
    covariance = avg_X2 - avg_means2
    covariance /= nk.sum()
    covariance.flat[:: len(covariance) + 1] += reg_covar
    return covariance


def _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar):
    """Estimate the diagonal covariance vectors."""
    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means ** 2
    avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
    return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar


def _estimate_gaussian_covariances_spherical(resp, X, nk, means, reg_covar):
    """Estimate the spherical variance values."""
    return _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar).mean(1)


def _estimate_cov_parameters(X, resp, reg_covar, covariance_type):
    """Estimate the Gaussian distribution parameters."""
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = {
        "full": _estimate_gaussian_covariances_full,
        "tied": _estimate_gaussian_covariances_tied,
        "diag": _estimate_gaussian_covariances_diag,
        "spherical": _estimate_gaussian_covariances_spherical,
    }[covariance_type](resp, X, nk, means, reg_covar)
    return torch.tensor(covariances, dtype=torch.float64)


def _eval_incomp_special(splited_X, observed_position, means, covariances):
    """Evaluate incomplete data using special covariance structure."""
    N = len(splited_X)
    all_means = means.repeat(N, 1, 1)
    all_means_observed = list(map(lambda x, y: x[:, y], all_means, observed_position))

    all_covariances = covariances.repeat(N, 1, 1, 1)

    def _split_covariance(all_covariance, observed_position):
        res = torch.stack(list(map(lambda x: x[observed_position][:, observed_position], all_covariance)))
        return res

    all_covariances_observed = list(map(_split_covariance, all_covariances, observed_position))

    mvn = list(map(_estimate_log_gaussian_prob, splited_X, all_means_observed, all_covariances_observed))
    return mvn


def _calculate_incomp_responsibility(X_comp, splited_X, observed_position, weight, times,
                                     pars_means, pars_covariance,
                                     f_mean, f_covariance,
                                     model_type='D'):
    N0, D = X_comp.shape
    N1 = len(splited_X)
    K = len(weight)

    if model_type == 'A':
        pass
    elif model_type == 'B':
        means = torch.stack(list(map(lambda k: f_mean(times, *k), pars_means)))
        covariances = pars_covariance
        if (len(covariances.shape) != 3):
            covariances = covariances.repeat(K, 1, 1)

        mvn_log0 = _estimate_log_gaussian_prob(X_comp, means, covariances)
        mvn0 = mvn_log0.T + torch.log(weight).expand(N0, K)
        log_resp0 = mvn0 - torch.logsumexp(mvn0, 1).expand(K, N0).T

        mvn_log1 = _eval_incomp_special(splited_X, observed_position, means, covariances)
        mvn1 = torch.stack(mvn_log1) + torch.log(weight).expand(N1, K)
        log_resp1 = mvn1 - torch.logsumexp(mvn1, 1).expand(K, N1).T
    elif model_type == 'C':
        pass
    elif model_type == 'D':
        means = torch.stack(list(map(lambda k: f_mean(times, *k), pars_means)))
        covariances = f_covariance(pars_covariance, D)
        if (len(covariances.shape) != 3):
            covariances = covariances.repeat(K, 1, 1)

        mvn_log0 = _estimate_log_gaussian_prob(X_comp, means, covariances)
        mvn0 = mvn_log0.T + torch.log(weight).expand(N0, K)
        log_resp0 = mvn0 - torch.logsumexp(mvn0, 1).expand(K, N0).T

        mvn_log1 = _eval_incomp_special(splited_X, observed_position, means, covariances)
        mvn1 = torch.stack(mvn_log1) + torch.log(weight).expand(N1, K)
        log_resp1 = mvn1 - torch.logsumexp(mvn1, 1).expand(K, N1).T

    return [log_resp0, log_resp1]


def MVN_normal_full(X, mean, covariance):
    mvn = torch.distributions.MultivariateNormal(mean, covariance).log_prob(X)
    return mvn


def loss_function(X_comp, splited_X, observed_position, times, pars_means, pars_covariance,
                  resp0, resp1, f_mean, f_covariance, weight, eps=2e-30, model_type='D'):
    N0, D = X_comp.shape
    N1 = len(splited_X)
    K = len(weight)

    if model_type == 'A':
        pass
    elif model_type == 'B':
        means = torch.stack(list(map(lambda k: f_mean(times, *k), pars_means)))
        covariances = pars_covariance
        if len(covariances.shape) != 3:
            covariances = f_covariance(pars_covariance, D).repeat(K, 1, 1)

        mvn_log0 = list(map(lambda x, y: MVN_normal_full(X_comp, x, y), means, covariances))
        mvn0 = torch.stack(mvn_log0).T + torch.log(weight).expand(N0, K)
        LLB0 = -torch.sum(resp0 * mvn0 - resp0 * torch.log(resp0 + eps))

        mvn_log1 = _eval_incomp_special(splited_X, observed_position, means, covariances)
        mvn1 = torch.stack(mvn_log1) + torch.log(weight).expand(N1, K)

        LLB1 = -torch.sum(resp1 * mvn1 - resp1 * torch.log(resp1 + eps))
        LLB = LLB0 + LLB1
    elif model_type == 'C':
        pass
    elif model_type == 'D':
        means = torch.stack(list(map(lambda k: f_mean(times, *k), pars_means)))
        covariances = f_covariance(pars_covariance, D)
        if len(covariances.shape) != 3:
            covariances = f_covariance(pars_covariance, D).repeat(K, 1, 1)

        mvn_log0 = list(map(lambda x, y: MVN_normal_full(X_comp, x, y), means, covariances))
        mvn0 = torch.stack(mvn_log0).T + torch.log(weight).expand(N0, K)
        LLB0 = -torch.sum(resp0 * mvn0 - resp0 * torch.log(resp0 + eps))

        mvn_log1 = _eval_incomp_special(splited_X, observed_position, means, covariances)
        mvn1 = torch.stack(mvn_log1) + torch.log(weight).expand(N1, K)

        LLB1 = -torch.sum(resp1 * mvn1 - resp1 * torch.log(resp1 + eps))
        LLB = LLB0 + LLB1

    return LLB


def loss_maximization(n_epochs, optimizer, X_comp, splited_X, observed_position, times,
                      pars_means, pars_covariance,
                      resp0, resp1,
                      f_mean, f_covariance,
                      weight, eps, model_type):
    for epoch in range(1, n_epochs + 1):
        LLB = loss_function(X_comp, splited_X, observed_position, times, pars_means, pars_covariance,
                            resp0, resp1, f_mean, f_covariance, weight, eps, model_type)
        optimizer.zero_grad()
        LLB.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print('Epoch %d, LLB %f' % (epoch, float(LLB)))

    return [pars_means, pars_covariance]


def split_data(X):
    """Split data into complete and incomplete parts."""
    N, D = X.shape
    tmp = np.argwhere(np.isnan(X))

    if len(tmp) == 0:
        return [N, X, 0, None, np.arange(N), None, 0, None, None]

    if any(np.unique(tmp[:, 0], return_counts=True)[1] == D):
        n2 = np.unique(tmp[:, 0])[np.where(np.unique(tmp[:, 0], return_counts=True)[1] == D)[0]]
        N2 = len(n2)
        X_empty = X[n2,]

        n1 = np.unique(tmp[:, 0])[np.where(np.unique(tmp[:, 0], return_counts=True)[1] < D)[0]]
        N1 = len(n1)
        X_incomp = X[n1, :]

        n0 = np.setdiff1d(np.arange(N), np.union1d(n1, n2))
        N0 = len(n0)
        X_comp = X[n0, :]
    else:
        n2 = None
        N2 = 0
        X_empty = None

        n1 = np.unique(tmp[:, 0])[np.where(np.unique(tmp[:, 0], return_counts=True)[1] < D)[0]]
        N1 = len(n1)
        X_incomp = X[n1, :]

        n0 = np.setdiff1d(np.arange(N), n1)
        N0 = len(n0)
        X_comp = X[n0, :]
    return [N0, X_comp, N1, X_incomp, n0, n1, N2, n2, X_empty]


def get_observed_position(X):
    """Get observed positions for incomplete data."""
    tmp = torch.argwhere(~torch.isnan(X))
    number_observed = tmp[:, 0].unique(return_counts=True)[1]
    observed_position = tmp[:, 1].split(number_observed.tolist())
    splited_X = list(map(lambda x, y: x[y], X, observed_position))
    return [splited_X, observed_position]


class FunClu:
    """Functional Clustering Model."""

    def __init__(self, K=3, seed=None,
                 mean_type='power_equation', covariance_type="SAD1_tied",
                 lr=1e-2, tol=1e-3, reg_covar=1e-6, max_iter=100,
                 init_params="kmeans", weights_init=None, means_init=None):
        self.elbo = None
        self.parameters = {}
        self.eps = np.finfo(float).eps
        self.is_fit = False
        self.reg_covar = reg_covar

        # core model method attribute
        index0 = MODEL_Collections['name_mean'].index(mean_type)
        self.f_mean = MODEL_Collections['functions_mean'][index0]
        self.f_mean_init = MODEL_Collections['functions_mean_init'][index0]
        self.f_mean_closed_form = MODEL_Collections['mean_closed_form'][index0]

        index1 = MODEL_Collections['name_cov'].index(covariance_type)
        self.f_covariance = MODEL_Collections['functions_cov'][index1]
        self.f_cov_closed_form = MODEL_Collections['cov_closed_form'][index1]

        self.mean_type = mean_type
        self.covariance_type = covariance_type

        self.reg_covar = reg_covar
        self.init_params = init_params

        if all([self.f_mean_closed_form, self.f_cov_closed_form]) == True:
            self.model_type = 'A'
        elif self.f_mean_closed_form == False and self.f_cov_closed_form == True:
            self.model_type = 'B'
        elif self.f_mean_closed_form == True and self.f_cov_closed_form == False:
            self.model_type = 'C'
        elif all([self.f_mean_closed_form, self.f_cov_closed_form]) == False:
            self.model_type = 'D'

        if seed:
            self.seed = seed
        else:
            self.seed = random.randint(0, 100000)

        self.hyperparameters = {
            "K": K,
            "seed": self.seed,
            "learning_rate": lr,
            "optimizer": 'Adam',
            "bounds_l_mu": None,
            "bounds_u_mu": None,
            "bounds_l_sig": None,
            "bounds_u_sig": None,
            "mean_type": mean_type,
            "covariance_type": covariance_type,
            "mean_function": self.f_mean,
            "covariance_function": self.f_covariance,
            "mean_init_function": self.f_mean_init,
            "covariance_init_function": None,
            "number_mean_pars": MODEL_Collections['n_pars_mean'][index0],
            "number_covariance_pars": MODEL_Collections['n_pars_cov'][index1]
        }

    def _pre_process_data(self):
        """Preprocess data with missing values."""
        inds = np.where(np.isnan(self.X))
        X_copy = self.X.copy()
        X_copy[inds] = np.take(np.nanmean(self.X, axis=0), inds[1])
        return X_copy

    def _initialize(self, X, times=None, trans_data=False):
        """Initialize the model parameters."""
        if (isinstance(X, pd.DataFrame)):
            self.data_colnames = list(X)
            self.data_rownames = list(X.index)
        else:
            self.data_colnames = None
            self.data_rownames = None

        # decide times
        try:
            times[0]
            self.times = times
            self.X = np.array(X)
        except Exception as e:
            times = np.log10(np.nansum(np.array(X), axis=0) + 1)
            self.order = np.argsort(times)
            times_new = times[self.order]

            self.X = np.array(X)[:, self.order]
            self.times = times_new

        if trans_data:
            self.X = np.log10(self.X + 1)
        else:
            pass

        self.N, self.D = self.X.shape
        K = self.hyperparameters["K"]

        # check none in dataset
        self.N0, self.X_comp, self.N1, self.X_incomp, self.n0, self.n1, self.N2, self.n2, self.X_empty = split_data(
            self.X)

        if (self.N1 == 0):
            self.contain_missing = False
            X_filled = self.X.copy()
        else:
            self.contain_missing = True
            X_filled = self._pre_process_data()

        if self.init_params == "kmeans":
            resp = np.zeros((self.N, K))
            label = (KMeans(n_clusters=K, n_init='auto', random_state=self.seed).fit(X_filled).labels_)
            resp[np.arange(self.N), label] = 1
        elif self.init_params == "random":
            np.random.seed(seed=self.seed)
            resp = np.random.uniform(size=(self.N, K))
            resp /= resp.sum(axis=0)[:, np.newaxis]

        # weight, center and covariance matrix
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        means = np.dot(resp.T, X_filled) / nk[:, np.newaxis]
        weight = (nk / self.N)

        self.parameters = {
            "weight": torch.tensor(weight, dtype=torch.float64),
            "resp": torch.tensor(resp, dtype=torch.float64),
            "means": torch.tensor(means, dtype=torch.float64),
            "covariances": None,
            "resp0": None,
            "resp1": None
        }

        if self.mean_type != 'None':
            pars_means_init = list(map(lambda k:
                                       get_mean_init(x=self.times,
                                                     y=k,
                                                     mean_func=self.f_mean_init,
                                                     n_mean_pars=self.hyperparameters['number_mean_pars']),
                                       means))

            pars_means_init = torch.tensor(np.array(pars_means_init), requires_grad=True, dtype=torch.float64)
            self.parameters["pars_means"] = pars_means_init
            self.parameters["means"] = list(map(lambda k: self.f_mean(torch.tensor(self.times, dtype=torch.float64), *k), pars_means_init))
            self.parameters["means"] = torch.vstack(self.parameters["means"]).detach()
        else:
            self.parameters["pars_means"] = torch.tensor(means, dtype=torch.float64)
            self.hyperparameters["number_mean_pars"] = K * self.D

        if self.f_cov_closed_form == True:
            self.parameters["pars_covariance"] = _estimate_cov_parameters(X_filled, resp, self.reg_covar,
                                                                          self.covariance_type)
            self.parameters["covariances"] = _estimate_cov_parameters(X_filled, resp, self.reg_covar,
                                                                      self.covariance_type)
            self.hyperparameters['number_covariance_pars'] = _n_cov_parameters(K, self.D, self.covariance_type)
        else:
            if 'full' in self.f_covariance.__name__:
                self.parameters["pars_covariance"] = torch.tensor(np.random.random(K * 2)).reshape(K, 2)
                self.parameters["pars_covariance"] = self.parameters["pars_covariance"].requires_grad_(True)
                self.parameters["covariances"] = self.f_covariance(self.parameters["pars_covariance"], self.D)
                self.hyperparameters['number_covariance_pars'] = self.parameters["pars_covariance"].size(0) * 2
            else:
                self.parameters["pars_covariance"] = torch.tensor(np.array([0.1, 0.1]), requires_grad=True, dtype=torch.float64)
                self.parameters["covariances"] = self.f_covariance(self.parameters["pars_covariance"], self.D).repeat(K, 1, 1)

        # convert dataset to tensor
        self.X = torch.tensor(self.X, requires_grad=False, dtype=torch.float64)
        self.times = torch.tensor(self.times, requires_grad=False, dtype=torch.float64)
        self.X_comp = torch.tensor(self.X_comp, requires_grad=False, dtype=torch.float64) if self.X_comp is not None else None
        self.X_incomp = torch.tensor(self.X_incomp, requires_grad=False, dtype=torch.float64) if self.X_incomp is not None else None

        if (self.N2 > 0):
            self.X_empty = torch.tensor(self.X_empty, requires_grad=False, dtype=torch.float64)

        if self.contain_missing:
            self.splited_X, self.observed_position = get_observed_position(self.X_incomp)
            self.missed_position = list(
                map(lambda n: torch.tensor(np.setdiff1d(list(range(self.D)), n.cpu().numpy()), dtype=torch.int64),
                    self.observed_position))
            self.position_rearranged = torch.stack(list(map(lambda x, y: torch.cat((x, y)), self.observed_position,
                                                            self.missed_position)))
            self.new_position = torch.stack(list(map(torch.argsort, self.position_rearranged)))
        else:
            self.missed_position, self.position_rearranged, self.new_position = None, None, None
            self.splited_X, self.observed_position = None, None

    def _likelihood_lower_bound(self):
        """Compute the LLB under the current parameters."""
        P, K = self.parameters, self.hyperparameters["K"]
        weight, resp, resp0, resp1, means, covariances = P["weight"], P["resp"], P["resp0"], P["resp1"], P["means"], P[
            "covariances"]
        pars_means, pars_covariance = P["pars_means"], P["pars_covariance"]

        if (self.contain_missing == True):
            likelihood_lower_bound = loss_function(self.X_comp, self.splited_X, self.observed_position,
                                                   self.times, pars_means, pars_covariance,
                                                   resp0, resp1,
                                                   self.f_mean, self.f_covariance,
                                                   weight, self.eps, self.model_type)
        else:
            means = torch.stack(list(map(lambda k: self.f_mean(self.times, *k), pars_means)))
            covariances = self.f_covariance(pars_covariance, self.D)
            mvn_log = _estimate_log_gaussian_prob(self.X, means, covariances)
            mvn = mvn_log.T + torch.log(weight).expand(self.N, K)
            likelihood_lower_bound = -torch.sum(resp * mvn - resp * torch.log(resp + self.eps))

        return likelihood_lower_bound

    def _E_step(self):
        """E step"""
        P, K = self.parameters, self.hyperparameters["K"]
        weight, resp0, resp1, means, covariances = P["weight"], P["resp0"], P["resp1"], P["means"], P["covariances"]
        pars_means, pars_covariance = P["pars_means"], P["pars_covariance"]

        if (self.contain_missing == True):
            log_resp0, log_resp1 = _calculate_incomp_responsibility(self.X_comp,
                                                                    self.splited_X,
                                                                    self.observed_position,
                                                                    weight,
                                                                    self.times,
                                                                    pars_means,
                                                                    pars_covariance,
                                                                    self.f_mean,
                                                                    self.f_covariance,
                                                                    self.model_type)
            P["resp0"] = torch.exp(log_resp0).detach()
            P["resp1"] = torch.exp(log_resp1).detach()
            P["resp"] = torch.vstack([P["resp0"], P["resp1"]])
        else:
            means = torch.stack(list(map(lambda k: self.f_mean(self.times, *k), pars_means)))
            covariances = self.f_covariance(pars_covariance, self.D)

            mvn_log = _estimate_log_gaussian_prob(self.X, means, covariances)
            mvn = mvn_log.T + torch.log(weight).expand(self.N, K)
            log_resp = mvn - torch.logsumexp(mvn, 1).expand(K, self.N).T
            P["resp"] = torch.exp(log_resp).detach()

    def _M_step(self, n_epochs=50):
        """M step"""
        P, K = self.parameters, self.hyperparameters["K"]
        resp, resp0, resp1 = P["resp"], P["resp0"], P["resp1"]
        weight, pars_means, pars_covariance = P["weight"], P["pars_means"], P["pars_covariance"]

        # update cluster priors
        nk = torch.sum(resp, axis=0)
        P["weight"] = (nk / self.N).detach() + torch.tensor(self.eps, dtype=torch.float64)

        if (self.contain_missing == True):
            if self.model_type == 'D':
                optimizer = torch.optim.Adam([pars_means, pars_covariance],
                                             lr=self.hyperparameters["learning_rate"])
                pars_means, pars_covariance = loss_maximization(n_epochs,
                                                                optimizer,
                                                                self.X_comp,
                                                                self.splited_X,
                                                                self.observed_position,
                                                                self.times,
                                                                pars_means,
                                                                pars_covariance,
                                                                resp0,
                                                                resp1,
                                                                self.f_mean,
                                                                self.f_covariance,
                                                                weight,
                                                                eps=self.eps,
                                                                model_type=self.model_type)
            elif self.model_type == 'B':
                optimizer = torch.optim.Adam([pars_means, pars_covariance],
                                             lr=self.hyperparameters["learning_rate"])
                pars_means, pars_covariance = loss_maximization(n_epochs,
                                                                optimizer,
                                                                self.X_comp,
                                                                self.splited_X,
                                                                self.observed_position,
                                                                self.times,
                                                                pars_means,
                                                                pars_covariance,
                                                                resp0,
                                                                resp1,
                                                                self.f_mean,
                                                                self.f_covariance,
                                                                weight,
                                                                eps=self.eps,
                                                                model_type=self.model_type)
        else:
            if self.model_type == 'D':
                optimizer = torch.optim.Adam([pars_means, pars_covariance],
                                             lr=self.hyperparameters["learning_rate"])
                for epoch in range(1, n_epochs + 1):
                    means = torch.stack(list(map(lambda k: self.f_mean(self.times, *k), pars_means)))
                    covariances = self.f_covariance(pars_covariance, self.D)

                    mvn_log = _estimate_log_gaussian_prob(self.X, means, covariances)
                    mvn = mvn_log.T + torch.log(weight).expand(self.N, K)
                    LLB = -torch.sum(resp * mvn - resp * torch.log(resp + self.eps))
                    optimizer.zero_grad()
                    LLB.backward()
                    optimizer.step()
            elif self.model_type == 'B':
                optimizer = torch.optim.Adam([pars_means, pars_covariance],
                                             lr=self.hyperparameters["learning_rate"])
                for epoch in range(1, n_epochs + 1):
                    means = torch.stack(list(map(lambda k: self.f_mean(self.times, *k), pars_means)))
                    covariances = pars_covariance
                    if len(covariances.shape) != 3:
                        covariances = covariances.repeat(K, 1, 1)

                    mvn_log = _estimate_log_gaussian_prob(self.X, means, covariances)
                    mvn = mvn_log.T + torch.log(weight).expand(self.N, K)
                    LLB = -torch.sum(resp * mvn - resp * torch.log(resp + self.eps))
                    optimizer.zero_grad()
                    LLB.backward()
                    optimizer.step()

        P["pars_means"], P["pars_covariance"] = pars_means, pars_covariance

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        K = self.parameters['pars_means'].shape[0]
        n_features = self.D
        cov_params = self.hyperparameters['number_covariance_pars']

        if self.covariance_type == "full":
            cov_params = K * n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "diag":
            cov_params = K * n_features
        elif self.covariance_type == "tied":
            cov_params = n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "spherical":
            cov_params = K

        mean_params = self.hyperparameters['number_mean_pars'] * K
        if mean_params == 0:
            mean_params = n_features * K

        return int(cov_params + mean_params + K - 1)

    def fit(self, X, times=None, trans_data=False, max_iter=50, tol=1e-3, verbose=True, verbose_interval=1, m_step_epochs=50):
        """Fit the model."""
        prev_vlb = -np.inf
        self._initialize(X, times, trans_data)

        for _iter in range(max_iter):
            try:
                self._E_step()
                self._M_step(n_epochs=m_step_epochs)

                vlb = self._likelihood_lower_bound().detach()

                if verbose:
                    if _iter % verbose_interval == 0:
                        print(f"iter = {_iter + 1}. log-likelihood lower bound = {vlb}")

                converged = _iter > 0 and torch.abs(vlb - prev_vlb) <= tol
                if torch.isnan(vlb) or converged:
                    break

                prev_vlb = vlb

            except torch.linalg.LinAlgError:
                print("Singular matrix: components collapsed")
                return -1

        # update parameters
        means = torch.stack(list(map(lambda k: self.f_mean(self.times, *k), self.parameters['pars_means'])))
        self.parameters["means"] = means

        covariances = self.f_covariance(self.parameters["pars_covariance"], self.D)
        if (len(covariances.shape) != 3):
            covariances = covariances.repeat(self.hyperparameters["K"], 1, 1)
        self.parameters["covariances"] = covariances

        # update hyperparameters
        self.elbo = vlb
        self.is_fit = True
        self.hyperparameters['BIC'] = self.bic()
        self.hyperparameters['max_iter'] = max_iter
        self.hyperparameters['tol'] = tol

        return 0

    def bic(self):
        """Bayesian information criterion for the current model."""
        return 2 * self._likelihood_lower_bound().detach().cpu().numpy() + self._n_parameters() * np.log(self.N * self.D)
