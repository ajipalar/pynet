from collections import namedtuple
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import pandas as pd
from functools import partial
import scipy as sp
import scipy.stats
import pyext.src.pynet_rng as rng
import pyext.src.matrix as mat
import pyext.src.stats as stats

# Flags


def check_cov(m):
    assert np.alltrue(m[np.diag_indices(len(m))] > 0), f"fail : diag"
    assert mat.is_positive_definite(m), f"fail pos"


def scatter_plot(x, y, title=None, N=1000, xy=None):
    assert len(x) == len(y)
    N = len(x)
    fig, axs = plt.subplots()
    if not title:
        title = f"Normal random variates rho {xy}\nN={N}"
    plt.plot(x, y, "k.")
    plt.title(f"Normal random variates rho {xy}\nN={N}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim((-4, 4))
    plt.xlim((-3, 3))
    ax = fig.gca()
    ax.vlines(np.mean(x), ymin=-5, ymax=5)
    ax.hlines(np.mean(y), xmin=-3, xmax=3)


def margins_plot(x, y, title=None):
    assert len(x) == len(y)
    N = len(x)
    fig, axs = plt.subplots(2, 2)
    if not title:
        title = f"Normal random variates rho {xy}\nN={N}"
    ax = axs[0, 0]
    ax.plot(x, y, "k.")
    ax.set_title(f"Normal random variates rho {xy}\nN={N}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim((-4, 4))
    ax.set_xlim((-3, 3))
    ax.vlines(np.mean(x), ymin=-5, ymax=5)
    ax.hlines(np.mean(y), xmin=-3, xmax=3)

    def helper(ax, arg, color="k", bins=20):
        ax.hist(arg, color=color, bins=bins)

    kwargs = {"color": "k", "bins": 100}
    ax = axs[0, 1]

    # transform = Affine2D().rotate_deg(90)
    # plot_extents = -5, 5, 0, 10
    # h = floating_axes.GridHelperCurveLinear(transform, plot_extents)
    # ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=h)
    helper(ax, x, **kwargs)
    fig.add_subplot(ax)

    ax = axs[1, 0]
    helper(ax, y, **kwargs)


def rprior(key, V, n=2, p=2):
    return rng.wishart(key, V=V, n=n, p=p)


def rprior_pred(key, V, n, p, n_replicates):
    keys = jax.random.split(key, 2)
    S = rprior(keys[0], V=V, n=n, p=p)
    mean = jnp.zeros(p)
    return jax.random.multivariate_normal(keys[1], mean, cov=S, shape=(n_replicates,))


def get_prior_pred(key, V, n_replicates=3, n_samples=100, n=2, p=2):
    Val = namedtuple("Val", "key data")
    data = jnp.zeros((n_samples, n_replicates, p))

    def body(i, val):
        # N x p
        key, k1 = jax.random.split(val.key)
        prior_pred = rprior_pred(k1, V=V, n=n, p=p, n_replicates=n_replicates)
        data = val.data.at[i].set(prior_pred)
        return Val(key, data)

    init = Val(key, data)
    val = jax.lax.fori_loop(0, n_samples, body, init)
    return val


def randPOSDEFMAT(key, p):
    A = jax.random.uniform(key, shape=(p, p))
    A = 0.5 * (A + A.T)
    A = A + jnp.eye(p) * p
    return A


def quad_plot_prelude(prior_mat_stat_df, ground_truth, n, n_samples, font_rc, **kwargs):
    Kstats = get_precision_matrix_stats(ground_truth, n=n)
    names = ["rowsum", "medians", "vars", "means", "mins", "maxs", "absdets"]

    Kstats = {names[i]: Kstats[i] for i in range(len(Kstats))}
    prior_mat_stat_df = prior_mat_stat_df[["means", "vars", "mins", "maxs"]]

    return Kstats, names, prior_mat_stat_df


def helper_vline_hist(
    ax, vx, ymin, ymax, vals, vlabel, hlabel, vcolor, hcolor, bins, ylabel, xlabel
):
    ax.set_xlabel(xlabel)
    ax.hist(vals, bins=bins, label=hlabel, facecolor=hcolor)
    ax.set_ylabel(ylabel)
    ax.vlines(ymin=ymin, ymax=ymax, x=vx, color=vcolor, label=vlabel)


def quad_plot(
    prior_mat_stat_df,
    ground_truth,
    n,
    n_samples,
    font_rc,
    p,
    suptitle: str = None,
    **kwargs,
):
    Kstats = get_precision_matrix_stats(ground_truth, n=n, p=p)
    names = ["rowsum", "medians", "vars", "means", "mins", "maxs", "absdets"]

    Kstats = {names[i]: Kstats[i] for i in range(len(Kstats))}
    prior_mat_stat_df = prior_mat_stat_df[["means", "vars", "mins", "maxs"]]
    scale = 8
    bins = 30
    w = 1 * scale
    h = 1 * scale
    cmap1 = "CMRmap"  # "nipy_spectral" #"CMRmap"\
    cmap2 = "nipy_spectral"
    facecolor = "steelblue"  # "pink"#"cornflowerblue"
    vlinecolor = "darkorange"  # "aquamarine"#"orange"
    cbar_scale = 0.35

    fig, axs = plt.subplots(2, 2, layout="constrained")

    locator = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}

    for i in range(4):
        ax = axs[locator[i]]
        col = prior_mat_stat_df.iloc[:, i]
        if i == 0:
            col = n * col
        xlabel = f"n-{col}" f"{col.name}"
        ax.set_xlabel(f"{col.name}")
        label = f"prior" if i == 3 else None
        ax.hist(col.values, bins=bins, label=label, facecolor=facecolor)
        ax.set_ylabel("Frequency")
        ylim = ax.get_ylim()

        label = f"Ground Truth" if i == 3 else None
        ax.vlines(
            ymin=0, ymax=ylim[1], x=Kstats[col.name], color=vlinecolor, label=label
        )
    if suptitle:
        plt.suptitle(suptitle)
    else:
        plt.suptitle(f"N={n_samples}")
    plt.rc("font", **font_rc)
    fig.set_figheight(h)
    fig.set_figwidth(w)
    fig.legend()
    plt.show()


# +
# What is the largest matrix this will work on?


def try_sampling(key, p):
    print(f"Trying {p}")
    n = p
    cov_prior = jnp.eye(p)
    n_replicates = 3
    n_samples = 100
    kwargs = {
        "p": p,
        "V": cov_prior,
        "n_replicates": n_replicates,
        "n_samples": n_samples,
        "n": n,
    }

    val = jax.jit(partial(get_prior_pred, **kwargs))(key)
    print(f"Done! {val.data.shape}")


def ccscatter(x, y):
    rho = np.corrcoef(x.values, y.values)
    pearson = rho[0, 1]
    pearson = np.round(pearson, decimals=3)
    plt.scatter(x, y, label=f"rho {pearson}")
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.legend()
    plt.show()


def simulate_from_prior(key, nu, K_0):
    return rng.wishart(key, K_0, nu, len(K_0))


def sample_from_prior(key, nu, p, n_samples, K_0):
    Val = namedtuple("Val", "keys samples")
    samples = jnp.zeros((n_samples, p, p))
    keys = jax.random.split(key, num=n_samples)

    init = Val(keys, samples)

    def body(i, val):
        K = simulate_from_prior(val.keys[i], nu, K_0)
        samples = val.samples.at[i].set(K)
        val = Val(val.keys, samples)
        return val

    val = jax.lax.fori_loop(0, n_samples, body, init)
    return val


def get_precision_matrix_stats(K, n, p=64):
    Stats = namedtuple("Stats", "rowsum medians means vars mins maxs absdets")
    assert K.shape == (p, p)

    return Stats(
        [np.sum(x) for x in K],
        np.median(K),
        np.mean(K),
        np.var(K),
        np.min(K),
        np.max(K),
        np.abs(sp.linalg.det(K)),
    )


def df_from_stats(stats, n):
    rowsum_s = np.array([i.rowsum for i in stats])
    med_s = np.array([i.medians for i in stats])
    var_s = np.array([i.vars for i in stats])
    mean_s = np.array([i.means for i in stats])
    min_s = np.array([i.mins for i in stats])
    max_s = np.array([i.maxs for i in stats])
    absdet = np.array([i.absdets for i in stats])

    rowsum_s = str(rowsum_s)

    data = {
        "rowsum": rowsum_s,
        "medians": med_s,
        "vars": var_s,
        "means": mean_s,
        "mins": min_s,
        "maxs": max_s,
        "absdets": absdet,
    }

    return pd.DataFrame(data=data)


def do_gridplot(
    exp,
    scale=6,
    w=1.5,
    h=1,
    bins=30,
    hcolor="steelblue",
    vcolor="darkorange",
    font_rc={"size": 16, "family": "sans-serif"},
    check_finite=True,
    decomposition="eigh",
):

    w = w * scale
    h = h * scale

    if decomposition == "eigh":

        def decomp(x):
            return sp.linalg.eigh(x, eigvals_only=True, check_finite=check_finite)

    elif decomposition == "svd":

        def decomp(x):
            U, s, VH = sp.linalg.svd(x)
            return s

    elif decomposition == "prec":

        def decomp(x):
            return x[np.diag_indices(len(x))]

    ground_truth = decomp(exp.cov_inv)

    eigs = np.zeros((len(exp.samples.samples), exp.p))

    for i in range(len(exp.samples.samples)):
        eigs[i] = decomp(exp.samples.samples[i])

    assert len(exp.cov) % 2 == 0, f"rank cov is odd"
    m = int(np.sqrt(len(exp.cov)))
    fig, axs = plt.subplots(m, m, layout="constrained")
    fig.set_figheight(h)
    fig.set_figwidth(w)

    count = -1
    axs[0, 0].set_ylabel("Frequency")

    for i in range(m):
        for j in range(m):
            count += 1
            ax = axs[i, j]
            eigvals = eigs[:, count]
            vx = ground_truth[count]
            ymin = 0

            vlabel = f"K eigenvalues" if count == m * m - 1 else None
            hlabel = f"prior samples" if count == m * m - 1 else None

            # helper_vline_hist(ax, vx, ymin, ymax, eigvals, vlabel, hlabel,
            # vcolor, hcolor, bins, ylabel=None, xlabel=None)

            truth = ground_truth[count]
            ax.hist(eigvals, bins=bins, label=hlabel, facecolor=hcolor)
            ymax = ax.get_ylim()[1]
            ax.vlines(x=truth, ymin=0, ymax=ymax, color=vcolor)

            if decomposition == "eigh":
                s = "\u03BB"

            elif decomposition == "svd":
                s = "\u03C3"

            elif decomposition == "prec":
                s = f"p"

            SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
            s = s + str(count + 1)
            s = s.translate(SUB)
            ax.set_xlabel(s)
            # ax.scatter(x, y)

            plt.suptitle(f"N={n_samples}")
    # plt.legend()
    plt.show()


def ground_truth_pair_plot(
    A,
    K,
    title1="",
    title2="",
    cmap1="nipy_spectral",
    cmap2="CMRmap",
    factor=1.0,
    vmin1=None,
    vmax1=None,
    vmin2=None,
    vmax2=None,
    overwrite_diags=True,
):
    font_rc = {"size": 14, "family": "sans-serif"}

    scale = 16
    w = 1 * scale
    h = 1 * scale
    cbar_scale = 0.35

    fig, axs = plt.subplots(
        1,
        2,
        gridspec_kw={"width_ratios": [1, 1], "height_ratios": [1]},
        layout="constrained",
    )

    plt.rc("font", **font_rc)
    fig.set_figheight(h)
    fig.set_figwidth(w)
    ax = axs[0]

    if not vmin1:
        vmin1 = np.min(A)
    if not vmax1:
        vmax1 = 2 * np.median(A)

    covim = ax.imshow(A, vmin=vmin1, vmax=vmax1, cmap=cmap1)
    fig.colorbar(covim, ax=ax, location="left", shrink=cbar_scale)
    ax.set_title(title1)
    # ax.legend()

    ax = axs[1]
    ax.set_title(title2)

    # if type(cmap2) == str:
    #     cmap2 = matplotlib.colors.Colormap(cmap2)
    K_plot = K.copy()

    if overwrite_diags:
        K_plot[np.diag_indices(len(K_plot))] = np.max(np.tril(K_plot, k=-1))

    if not vmin2:
        vmin2 = np.min(K_plot)
    if not vmax2:
        vmax2 = None
    precim = ax.imshow(K_plot, vmin=vmin2, vmax=vmax2, cmap=cmap2)

    # bounds = [-1/factor, 1/factor]
    # cnorm = matplotlib.colors.BoundaryNorm(bounds, cmap2.N)

    fig.colorbar(precim, ax=ax, location="right", shrink=cbar_scale)
    # plt.suptitle(, y=0.75)
    return fig, axs
