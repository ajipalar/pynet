from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#Import my modules

from score import mode, normal_pdf, parabola, ull_normal
def plot_density(length, mu, sig):

    xnorm = np.arange(-length, length)
    ynorm = normal_pdf(xnorm, mu, sig)
    plt.figure(figsize=(8, 6))
    plt.subplot(121)
    plt.title(f'Univariate Gaussain mu{mu}, sigma{sig}')
    plt.xlabel('x')
    plt.ylabel('P(x==mu)')
    plt.scatter(xnorm, ynorm)
    plt.subplot(122)
    plt.title(f'Univariate log Guassian density')
    plt.xlabel('x')
    plt.ylabel('ln P(x==mu)')
    plt.scatter(xnorm, np.log(ynorm))
    plt.tight_layout()
    plt.show()


def MCMC_chain_plot(chain, y):
    steps = len(chain)
    fig, (a1, a2) = plt.subplots(nrows=1, ncols=2)
    plt.suptitle('MCMC-MH Sampling a Guassian')
    a1.plot(chain)
    a1.set_title(f'steps={steps}')
    a1.legend(['mu', 'sigma'])
    a2.set_title('Data y')
    pretty_std = np.round(np.std(y), decimals=2)
    pretty_mu = np.round(np.mean(y), decimals=2)
    a2.hist(y)
    a2.text(0.8, 0.6, 
            f'sample mean:{pretty_mu}\nsample std{pretty_std}', 
            transform=a2.transAxes)
    plt.show()

def chain_summary(ax, title, chain, tx, ty):
    ax.hist(chain)
    ax.set_title(title)
    med = np.round(np.median(chain))
    std = np.round(np.std(chain))
    ax.text(tx, ty, f'med:{med}\nstd{std}', transform=ax.transAxes)

def summary_wrapper(chain, mu, sig):
    fig, (a1, a2) = plt.subplots(nrows=1, ncols=2)
    plt.suptitle(f'mu={mu}, sigma={sig}')
    chain_summary(a1, 'Estimate of mu', chain[:, 0], 0.7, 0.85)
    chain_summary(a2, 'Estimate of sigma', chain[:, 1], 0.6, 0.85)

def plot_parabola():
    mu = 11
    x = np.arange(-mu, 3 * mu , mu / 8)
    y = []
    for i in x:
        y.append(np.sum(parabola(i, mu)))
    plt.plot(x, y)


def plot_likelihood_func(mu, sig, length, width=0.5):
    pretty_sigma = np.round(sig, decimals=1)
    pretty_mu = np.round(mu, decimals=1)
    y = mu + np.random.randn(length) * sig

    #plt.style.use('ggplot')
    mu_guess = np.arange(mu - width * mu, mu +  width *mu, width * mu / 128)
    sig_guess = np.arange(sig - width * sig, sig +  width *sig, width * sig / 128)
    partial_ull_normal = partial(ull_normal, y = y, sigma = sig)
    partial_ull_sig = partial(ull_normal, y = y, mu = mu)


    fig, ax = plt.subplots(nrows=2, ncols=2)
    a1, a2, a3, a4 = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]
    mag = 10
    step = 0.5
    score = np.array(list(partial_ull_normal(mu = i) for i in mu_guess))
    a1.scatter(mu_guess, score)
    a1.set_title(f'ln P(mu | sig={pretty_sigma})')
    a1.set_ylabel(f'ul P(mu | sigma={pretty_sigma})')
    a1.set_xlabel(f'mu')
    a2.set_title(f'P(mu | sig={pretty_sigma})')
    a2.scatter(mu_guess, np.exp(score))

    score = np.array(list(partial_ull_sig(sigma = i) for i in sig_guess))

    a3.scatter(sig_guess, score)
    a3.set_title(f'ln P(sig | mu={pretty_mu})')
    a4.set_title(f'P(sig | mu={pretty_mu})')
    a4.scatter(sig_guess, np.exp(score))
    plt.tight_layout()
    plt.show()
