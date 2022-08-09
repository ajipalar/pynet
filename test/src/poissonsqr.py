"""
Tests for the poisson sqr module in pyext/src/poissonsqr
"""

from __future__ import print_function
import IMP
import IMP.test
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random
from jax import jit
import numpy as np
from hypothesis import given, settings, strategies as st
import hypothesis.extra.numpy as hnp
import hypothesis
from typing import Any, Callable
from functools import partial
import collections
import scipy as sp

Module = Any


def dev_remove_ith_entry__s(a, src: Module):
    f__j = src.remove_ith_entry__s(a=a)
    n = len(a)
    assert (a.ndim == 1) or (a.ndim == 2)

    # jit once
    jitf = jit(fun=f__j)
    jitf(arr=a, i=0)

    for i in range(len(a)):
        s1 = slice(1, i)
        s2 = slice(0, i)
        s3 = slice(i + 1, n)
        s4 = slice(i, n - 1)
        s5 = slice(0, n - 1)
        s6 = slice(0, n)
        if a.ndim == 2:
            s1 = (slice(0, n), s1)
            s2 = (slice(0, n), s2)
            s3 = (slice(0, n), s3)
            s4 = (slice(0, n), s4)
            s5 = (slice(0, n), s5)
            s6 = (slice(0, n), s6)

        out = f__j(arr=a, i=i)
        jout = jitf(arr=a, i=i)
        out = np.array(out)
        jout = np.array(jout)

        np.testing.assert_almost_equal(actual=out, desired=jout)
        if i == 0:
            # np.testing.assert_almost_equal(a[1:i], out[0:i])
            np.testing.assert_almost_equal(actual=a[s1], desired=out[s2])
        if 0 < i <= n:
            # np.testing.assert_almost_equal(a[0:i], out[0:i])
            np.testing.assert_almost_equal(actual=a[s2], desired=out[s2])
            # np.testing.assert_almost_equal(a[i+1:n], out[i:n-1])
            np.testing.assert_almost_equal(actual=a[s3], desired=out[s4])
        if i == n:
            # np.testing.assert_almost_equal(a[0:n-1], out[:])
            np.testing.assert_almost_equal(actual=a[s5], desired=out[s6])


def remove_ith_entry__s_vs_value(src: Module, d1=100, d2=10):
    """Tests the removal of the ith entry from a vector and a matrix
    where answers are known"""
    test_dtype = jnp.float32

    a1d = jnp.arange(start=d1, dtype=test_dtype)
    a2d = jnp.arange(start=d2 * d2, dtype=test_dtype).reshape((d2, d2))

    f1d__j = src.remove_ith_entry__s(a=a1d)
    f2d__j = src.remove_ith_entry__s(a=a2d)

    j1d = jit(fun=f1d__j)
    j2d = jit(fun=f2d__j)

    # jit compile once
    j1d(arr=a1d, i=0)
    j2d(arr=a2d, i=0)

    del f1d__j
    del f2d__j

    for i in range(d1):
        a1d_min_i = np.array(j1d(a1d, i))
        assert a1d_min_i.shape == (d1 - 1,)
        if i == 0:
            assert a1d[0] == 0
            assert a1d_min_i[0] == 1

            ref = a1d[1:d1]
            pred = a1d_min_i
            np.testing.assert_almost_equal(actual=ref, desired=pred)
        elif 0 < i < d1 - 1:
            ref1 = a1d[0:i]
            ref2 = a1d[i + 1 : d1]
            pred1 = a1d_min_i[0:i]
            pred2 = a1d_min_i[i : d1 - 1]
            np.testing.assert_array_almost_equal(x=ref1, y=pred1)
            np.testing.assert_array_almost_equal(x=ref2, y=pred2)
        # i==d1
        else:
            ref1 = a1d[0 : d1 - 1]
            pred1 = a1d_min_i
            np.testing.assert_almost_equal(actual=ref1, desired=pred1)

    for i in range(d2):
        a2d_min_i = np.array(j2d(a2d, i))
        assert a2d_min_i.shape == (d2, d2 - 1), f"{a2d_min_i.shape}"
        if i == 0:
            ref = a2d[:, 1:d2]
            pred = a2d_min_i
            np.testing.assert_almost_equal(actual=pred, desired=ref)

        elif 0 < i < d2 - 1:
            ref1 = a2d[:, 0:i]
            ref2 = a2d[:, i + 1 : d2]
            pred1 = a2d_min_i[:, 0:i]
            pred2 = a2d_min_i[:, i : d2 - 1]
            np.testing.assert_array_almost_equal(x=ref1, y=pred1)
            np.testing.assert_array_almost_equal(x=ref2, y=pred2)
        # i==d2
        else:
            ref1 = a2d[:, 0 : d2 - 1]
            pred1 = a2d_min_i
            np.testing.assert_almost_equal(actual=ref1, desired=pred1)


def get_exponent__j_is_jittable(theta, phi, x, src):
    ulog_score__j = src.get_exponent__s(theta=theta, phi=phi, x=x)
    jf = jit(fun=ulog_score__j)
    jf(theta=theta, phi=phi, x=x, i=0)


def helper_init_state(key, n: int, d: int):
    k1, k2, k3 = jax.random.split(key=key, num = 3)

    X = jax.random.poisson(key=k1, lam=11, shape=[d, n])
    theta = jax.random.normal(key=k2, shape=[d])
    phi = jax.random.normal(key=k3, shape=[d, d])
    return X, theta, phi


def get_eta2__j_values(src: Module, d : int = 3, xscale: float = 4.0, thetascale: float=1.0, phiscale:float =1.0):

    result = (np.ones(d - 1) * phiscale) @ (np.sqrt(np.ones(d - 1) * xscale))

    x = np.ones(shape=d, dtype=jnp.int32) * xscale
    theta = np.ones(shape=d) * thetascale
    phi = np.ones(shape=(d, d)) * phiscale

    get_eta2__j = src.get_eta2__s(theta, phi, x)

    a = np.array(get_eta2__j(theta, phi, x, 0))
    t1: float = thetascale
    t2: float = 2 * (result)
    eta2 = t1 + t2
    eta2 = np.array(eta2)
    np.testing.assert_almost_equal(actual=eta2, desired=a)


def get_exponent__j_values(
    src: Module,
    decimal: int,
    d: int =3,
    xscale: float =4.0,
    thetascale: float =1.0,
    phiscale: float =1.0,
    test_dtype=jnp.int32,
):

    x = jnp.ones(shape=d, dtype=jnp.int32) * xscale
    theta = jnp.ones(shape=d) * thetascale
    phi = jnp.ones(shape=(d, d)) * phiscale

    # i = 0
    # phi[i, i] * x[i]
    t1: float = phiscale * xscale
    # theta[i] + 2 * rm_i(phi[:, i], i) @ jnp.sqrt(rm_i(x, i))
    t2: float = (
        thetascale
        + 2 * ((np.ones(d - 1) * phiscale) @ (np.sqrt(np.ones(d - 1) * xscale)))
    ) * np.sqrt(xscale)
    t3: float = np.log(4 * 3 * 2 * 1)

    a: float  = t1 + t2 - t3

    assert theta.shape == (d,)
    assert phi.shape == (d, d)
    assert x.shape == (d,)

    get_exponent__j = src.get_exponent__s(theta=theta, phi=phi, x=x)
    b = get_exponent__j(theta=theta, phi=phi, x=x, i=0)
    a = np.array(a)
    b = np.array(b)
    np.testing.assert_almost_equal(actual=a, desired=b, decimal=decimal)


def erf_taylor_approx__j(z: complex, src: Module):
    erf = sp.special.erf
    erf_taylor__j = src.erf_taylor_approx__j

    a = np.array(erf(z))
    b = np.array(erf_taylor_approx__j)


def T1_nsteps_mh__s_normal(key, nsteps: int, d: int, src: Module, theta, phi):
    """Tests that theta and phi are the correct shape for the n steps MH algorithm"""

    def scoref(theta, phi):
        return theta @ phi[:, 0]

    T__j = src.T1_nsteps_mh__s(f=scoref, nsteps=nsteps, d=d)
    jT = jax.jit(T__j)

    theta, phi = jT(key=key, theta=theta, phi=phi)

    assert theta.shape == (d,)
    assert phi.shape == (d, d)


def T1_nsteps_mh__univariate_probabilstic_test(key, mu1: float, mu2: float, sig1: float, sig2: float, nsteps: int, src: Module):
    """Test the nsteps MH algorithm against two univariate normal distributions
    the dimension = 2"""

    d = 2
    k1, k2, k3 = jax.random.split(key=key, num=3)
    theta = jax.random.normal(key=k1, shape=[d])
    phi = jax.random.normal(key=k2, shape=[d, d])

    def scoref(theta, phi, mu1: float =mu1, sig1: float =sig1, mu2: float=mu2, sig2: float=sig2) -> float:
        """
        Args:
          theta:
            theta[0:2] is [mu1, mu2]

          phi:
            phi[0, 0:2] is [sigma1, sigma2]
        """

        a = jnp.sqrt((theta[0] - mu1) ** 2)
        b = jnp.sqrt((theta[1] - mu2) ** 2)
        c = a + b

        a = jnp.sqrt((phi[0, 0] - sig1) ** 2)
        b = jnp.sqrt((phi[0, 1] - sig2) ** 2)
        c = c + a + b
        return c

    # T = partial(src.T1_nsteps_mh__s(f=scoref, nsteps=nsteps, d=d))
    T = src.T1_nsteps_mh__s(f=scoref, nsteps=nsteps, d=d)
    jT = jax.jit(T)
    theta, phi = jT(key=k3, theta=theta, phi=phi)

    theta = np.array(theta)
    phi = np.array(phi)

    np.testing.assert_almost_equal(actual=theta[0], desired=mu1)  # Test fails here
    np.testing.assert_almost_equal(actual=theta[1], desired=mu2)
    np.testing.assert_almost_equal(actual=phi[0, 0], desired=sig1)
    np.testing.assert_almost_equal(actual=phi[0, 1], desired=sig2)


def T1_nsteps_mh_unorm_prob_check(key, mu1, src: Module, nsteps=int(1e6), decimals=1):
    """Apply a univariate normal pdf to the n-steps metropolis hastings algorithm"""
    scoref = lambda theta, phi: jax.scipy.stats.norm.pdf(theta[1], mu1)

    d = 2
    k1, k2, k3 = jax.random.split(key=key, num=3)
    theta = jax.random.normal(key=k1, shape=[d])
    phi = jax.random.normal(key=k2, shape=[d, d])
    T1__s = src.T1_nsteps_mh__s
    T = T1__s(f=scoref, nsteps=nsteps, d=d)
    jT = jax.jit(T)
    theta, phi = jT(key=k3, theta=theta, phi=phi)

    theta = np.array(theta)

    np.testing.assert_almost_equal(actual=theta[1], desired=mu1, decimal=decimals)


def T1_nsteps_mh__probabilistic_check(key, nsteps: int, src: Module, theta, phi):
    """Tests the n-steps MH algorithim against a bivariate normal distribution

    Args:
      theta:
        An array of means
      phi:
        A (d, d) symmetric covariance matrix
    """
    d = 2

    def bivariate_pdf():
        ...


def ais__s(
    key, d, nsamples: int, ninterpol: int, T: Callable, scoref: Callable, src: Module
):
    """Tests the ais function: failing"""
    test_function = src.ais__s
    ais__j = test_function(d=d, nsamples=nsamples, ninterpol=ninterpol, T=T, scoref=scoref)
    ais = jax.jit(fun=ais__j)
    ais(key=key)


def pretty_print_ss(k4, x: float, start_x: float, x_prime: float, xl: float, xr: float, u_prime: float, loop_break: bool):
    """Pretty prints the output of slice_sweep__j"""
    statement_width = 40
    pre: str = "  "
    a1: str = pre + f'key         :  {k4}'
    g1: str = " "*(statement_width - len(a1)  )
    a2: str = pre + f'start_x     :  {start_x}'

    a3: str = pre + f'x           :  {x}'
    g3: str = " "*(statement_width - len(a3))
    a4: str = pre + f'x_prime     :  {x_prime}'
    
    a5: str = pre + f'xl          :  {xl}'
    g5: str = " "*(statement_width -len(a5) )
    a6: str = pre + f'xr          :  {xr}'

    a7: str = pre + f'u_prime     :  {u_prime}'
    g7: str = " "*(statement_width -len(a7))
    a8: str = pre + f'loop_break  :  {loop_break}'

    out: str = '\n' + a1 + g1 + a2
    out += '\n' + a3 + g3 + a4
    out += '\n' + a5 + g5 + a6
    out += '\n' + a7 + g7 + a8
    print(out)

def slice_sweep__s_test_definition(key, f: Callable, w: float, start_x: float, src: Module):
    """Test definitions for a single slice sweep"""

    ss__j = partial(src.slice_sweep__s, pstar=f, w=w)
    ss = jax.jit(fun=ss__j)
    val = ss(key=key, x=start_x)
    k4, x, x_prime, xl, xr, u_prime, t, loop_break = val

    pretty_print_ss(k4, x, start_x, x_prime, xl, xr, u_prime, loop_break)



def slice_sweep_expected_value_test_definition(key, f, w, start_x, nsamples, src: Module):
    """Tests the expected value and variance of various distributions for the slice sweepinig
            algorithm"""

    samples = jnp.zeros(nsamples)
    pstar = f

    ss__j = partial(src.slice_sweep__s, pstar=pstar, w=w)
    ss = jax.jit(ss__j)

    x_prime = start_x

    for i in range(nsamples -1):
        key, k1 = jax.random.split(key=key)
        x_prime  = ss(key=k1, x=x_prime)[2]

        samples = samples.at[i].set(x_prime)


    pre = "  "
    print('\n' + pre + f'mean         :  {np.mean(samples)}')
    print(pre + f'var           :  {np.var(samples)}')

    ss = jax.jit(fun=ss__j)
    val = ss(key=k1, x=x_prime)
    k4, x, x_prime, xl, xr, u_prime, t, loop_break = val

    pretty_print_ss(k4, x, start_x, x_prime, xl, xr, u_prime, loop_break)

def slice_sweep__s_poisson_sqr_test_definition(key, theta, phi, xarr, x_start, d, w, src: Module):

    """Tests the slice sweep algorithm against the
       poisson sqr model. 

       params:
         key: jax PRNGKey
         theta: (d,) shape real array
         phi: (d,d) shape real symmetric array
         x: (d,) shape int32 array x[i] >= 0 for all i
         w: slice samling weight

    """

    # Define P*
    get_exp__j = src.get_exponent__s(theta=theta, phi=phi, x=xarr)
    pstar = partial(src.pstar, get_exponent__j = get_exp__j)
    pstar_kwargs = {"theta": theta, "phi": phi, "xarr": xarr, "i": 0}

    def update_pstar_kwargs(x, pstar_args, pstar_kwargs):
        d = pstar_kwargs
        theta = d['theta']
        phi  = d['phi']
        xarr = d['xarr']
        x_old = d['x']
        i_old = d['i']

        ...

    keys = jax.random.split(key, num=d)

    ss = partial(src.slice_sweep__s, pstar=pstar) 
    ss = jax.jit(ss)

    for i in range(d):
        val = ss(x=x_start, pstar=pstar, w=w, pstar_kwargs=pstar_kwargs)

def property_test_exponent_min_max2d(theta, phi, xarr, get_exponent, bound):

    exp0 = np.array(get_exponent(theta=theta, phi=phi, x=xarr, i=0))
    exp1 = np.array(get_exponent(theta=theta, phi=phi, x=xarr, i=1)) 

    assert  exp0 < bound, f"exp0 {exp0, bound}"
    assert  exp1 < bound, f"exp1 {exp1, bound}"


class PoissUnitTests(IMP.test.TestCase):
    """Base class for Poisson SQR unit tests"""

    src = None  # Derived class overides this
    rtol = None
    atol = None
    decimal = None
    kwds = None
    key = jax.random.PRNGKey(10)

    def test_slice_sweep__s_norm(self):
        """Slice sweep norm"""
        w = 1.
        pstar = partial(jax.scipy.stats.norm.pdf, 
                loc=7., 
                scale=2.) 
        slice_sweep__s_test_definition(key=self.key, 
            f=pstar, w=w, start_x=2., src=self.src)

    def test_slice_sweep__s_distant_initial_value(self):
        w = 1.
        pstar = partial(jax.scipy.stats.norm.pdf, loc=7., scale=2.) 
        slice_sweep__s_test_definition(key=self.key, f=pstar, w=w, start_x=-10., src=self.src)

        # fails because f(x0) = 0.
 #       slice_sweep__s_test_definition(key=self.key, f=pstar, w=w, start_x=-22., src=self.src)
        slice_sweep__s_test_definition(key=self.key, f=pstar, w=w, start_x=22., src=self.src)

    def test_slice_sweep__s_poisson(self):
        """Slice sweep poisson """
        w = 1.
        pstar = partial(jax.scipy.stats.poisson.pmf, mu=7.) 
        slice_sweep__s_test_definition(key=self.key, f=pstar, w=w, start_x=2., src=self.src)

    def test_slice_sweep__s_LW_norm(self):
        """Slice sweep norm low weight"""
        w = 0.1
        pstar = partial(jax.scipy.stats.norm.pdf, loc=7., scale=2.) 
        slice_sweep__s_test_definition(key=self.key, f=pstar, w=w, start_x=2., src=self.src)

    def test_slice_sweep__s_LW_poisson(self):
        """Slice sweep poisson low weight """
        w = 0.1
        pstar = partial(jax.scipy.stats.poisson.pmf, mu=7.) 
        slice_sweep__s_test_definition(key=self.key, f=pstar, w=w, start_x=2., src=self.src)


    def test_slice_sweep__s_HW_norm(self):
        """Slice sweep norm high weight """
        w = 10.
        pstar = partial(jax.scipy.stats.norm.pdf, loc=7., scale=2.) 
        slice_sweep__s_test_definition(key=self.key, f=pstar, w=w, start_x=2., src=self.src)


    def test_slice_sweep__s_HW_poisson(self):
        """Slice sweep poisson high weight """
        w = 10.
        pstar = partial(jax.scipy.stats.poisson.pmf, mu=7.) 
        slice_sweep__s_test_definition(key=self.key, f=pstar, w=w, start_x=2., src=self.src)

    def test_slice_sweep__s_expected_value_norm(self):
        """slice sweep expected value norm """

        nsamples = 99
        start_x = 2.
        pstar = partial(jax.scipy.stats.norm.pdf, loc=7., scale=2.) 
        w=1.
        slice_sweep_expected_value_test_definition(key=self.key, f=pstar, w=w, start_x=start_x, nsamples=nsamples, src=self.src)

    def test_slice_sweep__s_expected_value_poisson(self):
        """slice sweep expected value poisson """

        nsamples = 99
        start_x = 2.
        pstar = partial(jax.scipy.stats.poisson.pmf, mu=7.)
        w=1.
        slice_sweep_expected_value_test_definition(key=self.key, f=pstar, w=w, start_x=start_x, nsamples=nsamples, src=self.src)
    
    @IMP.test.skip
    def test_slice_sweep__s_sqr(self):

        slice_sweep__s_poisson_sqr_test_definition(key=self.key,
                theta=self.theta2d,
                phi=self.phi2d,
                xarr=self.x2d,
                x_start=1.,
                d=2,
                w=1.,
                src=self.src)

        
        


    # @IMP.test.skip  #dev
    def test_dev_remove_ith_entry(self):
        def run(key, shape):
            key, subkey = jax.random.split(key=key)
            m = jax.random.normal(key=key, shape=shape)
            dev_remove_ith_entry__s(a=m, src=self.src)
            return key

        key = jax.random.PRNGKey(seed=5)
        key = run(key=key, shape=(2,))
        key = run(key=key, shape=(3,))
        key = run(key=key, shape=(5,))
        key = run(key=key, shape=(6,))
        key = run(key=key, shape=(2, 2))
        key = run(key=key, shape=(3, 3))
        key = run(key=key, shape=(4, 4))
        # key = run(key, (111, 111))

    # @IMP.test.skip  #dev
    def test_dev_remove_ith_entry_vs_value(self):
        remove_ith_entry__s_vs_value(src=self.src)

    # @IMP.test.skip  #dev
    def test_get_ulog_score_is_jittable(self):
        d = 3
        n = 1
        X, theta, phi = helper_init_state(key=self.key, n=n, d=d)
        assert theta.shape == (d,)
        assert phi.shape == (d, d)
        assert X.shape == (d, n)
        x = X[:, 0]
        assert x.shape == (d,)
        # assert theta.dtype == np.float32
        # assert phi.dtype == np.float32
        # assert x.dtype == np.float32
        get_exponent__j_is_jittable(theta=theta, phi=phi, x=x, src=self.src)

    # @IMP.test.skip  #dev
    def test_logfactorial(self):

        factorial = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]
        lfactorial = np.log(factorial)
        for i, lf in enumerate(lfactorial):
            logfaci = self.src.logfactorial(n=i)
            np.testing.assert_almost_equal(actual=lf, desired=logfaci, decimal=self.decimal)

    # @IMP.test.skip  #dev
    def test_get_eta2__j(self):
        precision = 5
        n = 4
        d = 4
        X, theta, phi = helper_init_state(key=self.key, n=n, d=d)

        del n

        assert len(theta) == len(phi) == d

        x = X[:, 1]
        get_eta2__j = self.src.get_eta2__s(theta=theta, phi=phi, x=x)
        jf = jax.jit(fun=get_eta2__j)
        jf(theta=theta, phi=phi, x=x, i=0).block_until_ready()

        del X

        # i == 0
        a = np.array(get_eta2__j(theta=theta, phi=phi, x=x, i=0))
        b = theta[0] + 2 * (phi[:, 0][1:d] @ jnp.sqrt(x[1:d]))
        b = np.array(b)
        np.testing.assert_almost_equal(actual=a, desired=b, decimal=precision)

        del a
        del b

        for i in range(1, d - 1):
            tmp = np.zeros(d - 1)
            tmp[0:i] = phi[0:i, i]
            tmp[i : d - 1] = phi[i + 1 : d, i]
            b = tmp

            tmp = np.zeros(d - 1)
            tmp[0:i] = x[0:i]
            tmp[i : d - 1] = x[i + 1 : d]
            tmp = np.sqrt(tmp)

            a = np.array(get_eta2__j(theta=theta, phi=phi, x=x, i=i))
            b = np.array(theta[i]) + 2 * (b @ tmp)
            np.testing.assert_almost_equal(actual=a, desired=b, decimal=precision)

            del tmp
            del a
            del b
            del i

        # i == d-1
        a = np.array(get_eta2__j(theta=theta, phi=phi, x=x, i=(d - 1)))
        tmp = phi[0 : d - 1, d - 1]
        b = tmp
        tmp = jnp.sqrt(x[0 : d - 1])
        b = theta[d - 1] + 2 * (b @ tmp)
        b = np.array(b)
        np.testing.assert_almost_equal(actual=a, desired=b, decimal=precision)

        del a
        del b
        del tmp
        del precision
        del d
        del theta
        del phi
        del jf
        del get_eta2__j

    # @IMP.test.skip  #dev
    def test_get_eta2__j_values(self):
        get_eta2__j_values(src=self.src)

    # @IMP.test.skip  #dev
    def test_get_eta2__j_valuesd4(self):
        get_eta2__j_values(d=4, src=self.src)

    # @IMP.test.skip  #dev
    def test_get_exponent__j_values(self):
        DECIMALS = 5
        get_exponent__j_values(src=self.src, decimal=DECIMALS)

    # @IMP.test.skip  #dev
    def test_T1_nsteps_mh__s_normal(self):
        d = 2
        nsteps = 4
        theta = jnp.zeros(d).block_until_ready()
        phi = jnp.zeros((d, d)).block_until_ready()
        T1_nsteps_mh__s_normal(self.key, nsteps, d, self.src, theta, phi)
        d = 3
        theta = jnp.zeros(d).block_until_ready()
        phi = jnp.zeros((d, d)).block_until_ready()
        T1_nsteps_mh__s_normal(key=self.key, nsteps=nsteps, d=d, src=self.src, theta=theta, phi=phi)

    # @IMP.test.skip  #dev
    def test_ais__j_jittable(self):
        d=2
        nsamples = 10
        ninterpol = 5
        T = lambda a, b: (a, b)
        scoref = lambda x: x + 1

        ais__s(key=self.key, d=self.d, nsamples=nsamples, ninterpol=ninterpol, T=T, scoref=scoref, src=self.src)

    # @IMP.test.skip
    def test_T1_unormal_prob(self):
        nsteps = 10000
        mu1 = 0.3
        sig1 = 0.2
        mu2 = -0.7
        sig2 = 0.1

        T1_nsteps_mh__univariate_probabilstic_test(
            key=self.key, mu1=mu1, mu2=mu2, sig1=sig1, sig2=sig2, nsteps=nsteps, src=self.src
        )

    # @IMP.test.skip  #dev
    def test_T1_nsteps_mh_unorm_prob_check(self):
        key = jax.random.PRNGKey(self.rseed)
        mu1 = 123.3

        T1_nsteps_mh_unorm_prob_check(key=key, mu1=mu1, src=self.src)


class IsMatrixCompatible(IMP.test.TestCase):
    """Tests if functions are matrix compatible"""


class PoissPropTests(IMP.test.TestCase):
    """Base class for Poisson SQR Property tests"""

    @IMP.test.skip # dev
    @given(
        st.integers(min_value=2, max_value=13), st.integers(min_value=2, max_value=13)
    )
    @settings(deadline=None)
    def test_remove_ith_entry__s_vs_value(self, d1: int, d2: int):
        remove_ith_entry__s_vs_value(src=self.src, d1=d1, d2=d2)

    # @given(
    #        hnp.arrays(np.float32, (2,)), hnp.arrays(np.float32, (2, 2)), hnp.arrays(np.int32, (2,))
    #)

    @given( hnp.arrays(np.int32, (2,), elements=st.integers(min_value=0, max_value=100)))
    @settings(deadline=None) #, verbosity=hypothesis.Verbosity.verbose)
    def test_exponent_min_max(self, xarr):


        get_exponent = self.get_exp2d
        theta=self.theta2d
        phi=self.phi2d

        property_test_exponent_min_max2d(theta=theta, phi=phi, xarr=xarr, get_exponent=get_exponent, bound=1000)

    @given( hnp.arrays(np.float32, (2,), elements=st.floats()))
    def test_exponent_theta(self, theta):

        get_exponent = self.get_exp2d
        theta=self.theta2d
        phi=self.phi2d
        xarr = self.x2d
        property_test_exponent_min_max2d(theta=theta, phi=phi, xarr=xarr, get_exponent=get_exponent, bound=1000)





        

        
