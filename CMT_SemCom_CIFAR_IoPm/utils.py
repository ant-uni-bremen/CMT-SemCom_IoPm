import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
import torch.nn.functional as F


def train_density_ratio_estimator(x_nu, x_de, sample_size):
    # Randomly select a subset of samples from x_nu and x_de
    idx_nu = np.random.choice(len(x_nu), sample_size, replace=False)
    idx_de = np.random.choice(len(x_de), sample_size, replace=False)
    
    x_nu_subset = x_nu[idx_nu]
    x_de_subset = x_de[idx_de]
    shape_newx = x_nu_subset.shape
    # Combine samples and labels
    X = np.concatenate([x_nu_subset, x_de_subset])
    shape = X.shape
    y_nu = np.ones(x_nu_subset.shape[0])
    y_de = -np.ones(x_de_subset.shape[0])
    y = np.concatenate([y_nu, y_de])
    #y = np.concatenate([np.ones_like(x_nu), -np.ones_like(x_de)])
    shape_y = y.shape
    # Compute Gaussian kernel
    sigma = 3.0
    K = rbf_kernel(X, X, gamma=1/(2*sigma**2))
    # Train logistic regression model
    clf = LogisticRegression(C=1.0, solver='lbfgs')
    clf.fit(K, y)
    # Returning the trained parameters
    theta = clf.coef_.flatten()
    b = clf.intercept_[0]
    return theta, b, X


def estimate_density_ratio(new_samples, original_samples, theta, b):
    sigma = 3.0
    K_new = rbf_kernel(new_samples, original_samples, gamma=1/(2*sigma**2))
    DRE = K_new.dot(theta) + b
    return torch.tensor(DRE, dtype=torch.float32)

    
def awgn_channel(x):
    noise = torch.randn_like(x)
    return x + noise


def reparameterize(mu, ln_var):
    std = torch.exp(0.5 * ln_var)
    eps = torch.randn_like(std)
    c = mu + std * eps
    return c


def gaussian_kl_divergence(mu, ln_var, dim=1):
    return torch.sum(-0.5 * (1 + ln_var - mu.pow(2) - torch.exp(ln_var)), dim=dim)


def kl_log_uniform(alpha_squared):
    k1 = 0.63576
    k2 = 1.8732
    k3 = 1.48695
    KL_term = k1 * F.sigmoid(k2 + k3 * torch.log(alpha_squared)) - 0.5 * F.softplus(-1 * torch.log(alpha_squared)) - k1

    return - torch.sum(KL_term)


def sample_log_uniform_positive(shape, low_exp=-6, high_exp=3, base=2.0, device='cpu'):
    # Uniform in log-space between base^low_exp and base^high_exp
    u = torch.rand(shape, device=device)  # Uniform(0,1)
    log_min = low_exp * torch.log(torch.tensor(base, device=device))
    log_max = high_exp * torch.log(torch.tensor(base, device=device))
    log_sample = log_min + (log_max - log_min) * u
    return torch.exp(log_sample)



def sample_log_uniform_signed(shape, low_exp=-6, high_exp=3, base=2.0, device='cpu'):
    u = torch.rand(shape, device=device)  # Uniform in [0,1)
    log_min = low_exp * torch.log(torch.tensor(base, device=device))
    log_max = high_exp * torch.log(torch.tensor(base, device=device))
    log_mag = log_min + (log_max - log_min) * u
    sign = torch.sign(torch.randn(shape, device=device))
    return sign * torch.exp(log_mag)



#def sample_log_uniform_signed(shape, low_exp=-8, high_exp=8, base=2.0, device='cpu'):
#    u = torch.rand(shape, device=device)
#    log_min = low_exp * torch.log(torch.tensor(base, device=device))
#    log_max = high_exp * torch.log(torch.tensor(base, device=device))
#    log_mag = log_min + (log_max - log_min) * u
#    mag = torch.exp(log_mag)
#    sign = torch.where(torch.rand(shape, device=device) < 0.5, -1.0, 1.0)
#    return sign * mag

