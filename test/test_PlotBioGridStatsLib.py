from __future__ import print_function
import IMP.test
import IMP.algebra

try:
    import IMP.pynet
    import IMP.pynet.PlotBioGridStatsLib as nblib
except ModuleNotFoundError:
    import pyext.src.PlotBioGridStatsLib as nblib

from functools import partial
import io
import jax
import jax.numpy as jnp
import os
import math
import mygene
import numpy as np
import unittest
from pathlib import Path


def get_test_path():
    return Path("../devel/PlotBioGridStatsLib.py")


class TestMyGene(IMP.test.TestCase):
    """Test functionality that depends on the third party python module mygene
    used to query bioinformatic databases such as NCBI"""

    pass


class TestPoissonSQR(IMP.test.TestCase):
    """Test functionality related to Poisson Square Root Graphical Models
    as shown in Inouye 2016"""

    seed0 = 7
    key0 = jax.random.PRNGKey(seed0)
    p0 = 5
    phi0 = nblib.get_random_phi_matrix(key0, p0)

    theta, phi, X, p, n = nblib.dev_get_dev_state_poisson_sqr()

    def test__get_vec_minus_s(self):
        print("Running test__get_vec_minus_s")

        test_vec = jnp.arange(31) * 2.1

        vec_l = len(test_vec)

        b1 = False
        b2 = False

        for s in range(1, vec_l + 1):
            vec_minus_s = nblib._get_vec_minus_s(s, test_vec, vec_l)

            if s - 1 == 0:
                assert jnp.all(test_vec[1:vec_l] == vec_minus_s)
                b1 = True

            elif s - 1 < vec_l:
                assert jnp.all(test_vec[0] == vec_minus_s[0])
                assert test_vec[s - 1] not in vec_minus_s
                assert jnp.all(test_vec[s:vec_l] == vec_minus_s[s - 1 : vec_l - 1])
                assert jnp.all(test_vec[0 : s - 1] == vec_minus_s[0 : s - 1])
                assert jnp.all(test_vec[s:vec_l] == vec_minus_s[s - 1 : vec_l - 1])
                b2 = True

        assert b1
        assert b2
        print("End test__get_vec_minus_s")

    def test_matrix_minus_slice(self):
        print("Running test_matrix_minus_slice")
        poisson_lam = 8

        p = self.p0
        phi = self.phi0

        # jit compile the function
        slicef = nblib.get_matrix_col_minus_s
        slicef = partial(slicef, p=p)
        slicef = jax.jit(slicef)

        rtol = 1e-05
        is_close = lambda a, b: jnp.allclose(a, b, rtol=rtol)
        is_close2 = lambda a, b: np.allclose(a, b, rtol=rtol)

        branch1_executed = False
        branch2_executed = False

        for s in range(1, p + 1):
            col = phi[:, s - 1]
            sl = slicef(s, phi)
            # print(i, col, sl)
            i = s - 1
            if i == 0:
                branch1_executed = True
                a = col[1:p]
                b = sl
                # print(f'b1 {s} {a} {b} {col} {sl}')
                assert is_close(a, b)
                assert is_close2(a, b)

            else:
                branch2_executed = True
                a = col[0:i]
                b = sl[0:i]
                # print(f'b2a {s} {a} {b} {col} {sl}')
                assert is_close(a, b)
                assert is_close2(a, b)

                a = col[s:p]
                b = sl[i : len(sl)]
                # print(f'b2b {s} {a} {b} {col} {sl}')
                assert is_close(a, b)
                assert is_close2(a, b)

        assert branch1_executed
        assert branch2_executed
        print("End matrix minus s")

    def test_eta1_eta2_jittable(self):
        theta = self.theta
        phi = self.phi
        X = self.X
        p = self.p
        n = self.n

        for i in range(0, len(X)):
            x_i = X[:, i]
            jget_eta1 = jax.jit(nblib.get_eta1)
            jget_eta2 = partial(nblib.get_eta2, p=p)
            jget_eta2 = jax.jit(jget_eta2)
            for s in range(p + 1):
                print(jget_eta1(phi, s))
                print(jget_eta2(theta, phi, x_i, s))

    def test_Aexp_evaluation(self):
        key = self.key0
        theta = self.theta
        phi = self.phi
        X = self.X
        p = self.p
        n = self.n

        print("Run test_Aexp_evaluation")

        gamma = 0
        theta = theta * gamma
        phi = nblib.get_exp_random_phi(key, p)

        for i in range(len(X)):
            x_i = X[:, i]
            for s in range(1, p + 1):
                eta1 = nblib.get_eta1(phi, s)
                eta2 = nblib.get_eta2(theta, phi, x_i, s, p)

                a_exp = nblib.Aexp(eta1, eta2)
                z_exp = nblib.Zexp(eta1, eta2)

                rtols = [1e-01, 1e-02, 1e-03, 1e-04, 1e-05, 1e-06, 1e-07]
                for rtol in rtols:
                    try:
                        assert jnp.allclose(a_exp, jnp.log(z_exp), rtol=rtol)
                        assert jnp.allclose(jnp.exp(a_exp), z_exp, rtol=rtol)
                    except AssertionError:
                        print(i, rtol, a_exp, jnp.log(z_exp))
                        assert False
        print("End test_Aexp_evalution")

    def test_ll_exponential_sqr(self):
        print("Run test_ll_expoential_sqr")

        key = self.key0
        theta = self.theta
        phi = self.phi
        X = self.X
        p = self.p
        n = self.n

        gamma = 0
        theta = theta * gamma
        phi = nblib.get_exp_random_phi(key, p)
        Aexp = nblib.Aexp

        for i in range(len(X)):
            x_i = X[:, i]
            for s in range(1, p + 1):
                eta1 = nblib.get_eta1(phi, s)
                eta2 = nblib.get_eta2(theta, phi, x_i, s, p)

                ll = nblib.ll_exponential_sqr(x_i, eta1, eta2, p, Aexp)
                assert ll < 0

        print("End test_ll_expoential_sqr")

    def test_log_exponential_sqr(self):
        print("Run test_ll_expoential_sqr")

        key = self.key0
        theta = self.theta
        phi = self.phi
        X = self.X
        p = self.p
        n = self.n

        gamma = 0
        theta = theta * gamma
        phi = nblib.get_exp_random_phi(key, p)
        Aexp = nblib.Aexp

        for i in range(len(X)):
            x_i = X[:, i]
            for s in range(1, p + 1):
                eta1 = nblib.get_eta1(phi, s)
                eta2 = nblib.get_eta2(theta, phi, x_i, s, p)

                log_exp = nblib.log_exponential_sqr_node_conditional(
                    x_i, eta1, eta2, s, Aexp
                )
                assert log_exp < 0

        print("End test_ll_expoential_sqr")


class TestBiogridDataLoading(IMP.test.TestCase):
    """Test the loading of biogrid data into python structures"""

    pass

    @unittest.skip
    def test_ncbi_gene_id(self):
        """This test fails because it takes too long to run
        must reimplement the functionality"""

        print("Running test_ncbi_gene_id")
        get_ncbi_gene_id = nblib.get_ncbi_gene_id
        dpath = Path("data/biogrid/BIOGRID-ALL-4.4.206.tab3.txt")
        mg = mygene.MyGeneInfo()

        biogrid = nblib.prepare_biogrid(dpath)
        for i, (rname, row) in enumerate(biogrid.iterrows()):
            if i % 1000 == 0:
                print(f"testing {i}")
            genename_a = row["Official Symbol Interactor A"]
            genename_b = row["Official Symbol Interactor B"]

            biogrid_entrez_a = str(row["Entrez Gene Interactor A"])
            biogrid_entrez_b = str(row["Entrez Gene Interactor B"])

            ida = get_ncbi_gene_id(genename_a, mg)
            idb = get_ncbi_gene_id(genename_b, mg)

            # assert type(ida) == int
            # assert type(idb) == int

            ncbi_gene_id_a = str(ida)
            ncbi_gene_id_b = str(idb)

            try:
                assert biogrid_entrez_a == ncbi_gene_id_a
                assert biogrid_entrez_b == ncbi_gene_id_b
            except AssertionError:
                print("test exception at")
                print(f"\t{i} {biogrid_entrez_a} {ncbi_gene_id_a}")
                print(f"\t{biogrid_entrez_b} {ncbi_gene_id_b}")
                na = ncbi_gene_id_a
                nb = ncbi_gene_id_b
                ea = biogrid_entrez_a
                eb = biogrid_entrez_b

                print(f"\t{type(na)}, {type(ea)}")
                print(f"\t{type(nb)}, {type(eb)}")


class TestSpectralCountsDataLoading(IMP.test.TestCase):
    """Test the loading of spectral counts data into pyhton structures"""

    pass


if __name__ == "__main__":
    IMP.test.main()
