import torch

    
def awgn_channel(x):
    noise = torch.randn_like(x)
    return x + noise

def reparameterize(mu, ln_var):
    std = torch.exp(0.5 * ln_var)
    eps = torch.rand_like(std)
    c = mu + std * eps
    return c

def gaussian_kl_divergence(mu, ln_var, dim=1):
    return torch.sum(-0.5 * (1 + ln_var - mu.pow(2) - torch.exp(ln_var)), dim=dim)