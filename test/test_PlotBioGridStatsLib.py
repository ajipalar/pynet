from __future__ import print_function
import IMP.test
import IMP.algebra
try:
    import IMP.pynet
except ModuleNotFoundError:
    import devel.PlotBioGridStatsLib as nblib

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

    def test_magnitude(self):
        """Write the test cast, print statemetns ok"""
        pass

class TestPoissonSQR(IMP.test.TestCase):
    """Test functionality related to Poisson Square Root Graphical Models
       as shown in Inouye 2016"""

    def test_matrix_minus_slice(slicef):
        slicef = nblib.get_matrix_col_minus_s
        seed = 7
        key = jax.random.PRNGKey(seed)
        p = 5
        poisson_lam = 8
        phi = nblib.get_poisson_matrix(key, p, poisson_lam)
        
        jslice_neg_s = nblib.jbuild(slicef, **{'p':p})
    
        for i in range(13):
            print(i)
            col = phi[:, i]
            sl = jslice_neg_s(i, phi)
            #print(i, col, sl)
            if i == 0:
                #print(i, col, sl)
                assert jnp.all(col[1:p] == sl)
            elif i < len(phi):
                assert jnp.all(col[0:i] == sl[0:i])
                assert jnp.all(col[i+1:p] == sl[i:len(sl)])
            else:
                assert jnp.all(col[0:p-1] == sl)
    

class TestBiogridDataLoading(IMP.test.TestCase):
    """Test the loading of biogrid data into python structures"""
    
        
    def test_magnitude(self):
        """Write the test cast, print statemetns ok"""
        pass

    @unittest.skip
    def test_ncbi_gene_id(self):
        """This test fails because it takes too long to run
           must reimplement the functionality"""
        get_ncbi_gene_id = nblib.get_ncbi_gene_id
        dpath = Path("data/biogrid/BIOGRID-ALL-4.4.206.tab3.txt")
        mg = mygene.MyGeneInfo()

        biogrid = nblib.prepare_biogrid(dpath) 
        for i, (rname, row) in enumerate(biogrid.iterrows()):
            if i % 1000 == 0 : print(f'testing {i}')
            genename_a = row['Official Symbol Interactor A']
            genename_b = row['Official Symbol Interactor B']
            
            biogrid_entrez_a = str(row['Entrez Gene Interactor A'])
            biogrid_entrez_b = str(row['Entrez Gene Interactor B'])

            ida = get_ncbi_gene_id(genename_a, mg)
            idb = get_ncbi_gene_id(genename_b, mg)

            #assert type(ida) == int
            #assert type(idb) == int
            
            ncbi_gene_id_a = str(ida)
            ncbi_gene_id_b = str(idb)
            
            try:
                assert biogrid_entrez_a == ncbi_gene_id_a
                assert biogrid_entrez_b == ncbi_gene_id_b
            except AssertionError:
                print('test exception at') 
                print(f'\t{i} {biogrid_entrez_a} {ncbi_gene_id_a}')
                print(f'\t{biogrid_entrez_b} {ncbi_gene_id_b}')
                na = ncbi_gene_id_a
                nb = ncbi_gene_id_b
                ea = biogrid_entrez_a
                eb = biogrid_entrez_b
                
                print(f'\t{type(na)}, {type(ea)}')
                print(f'\t{type(nb)}, {type(eb)}')
    
class TestSpectralCountsDataLoading(IMP.test.TestCase):
    """Test the loading of spectral counts data into pyhton structures"""

    def test_magnitude(self):
        """Write the test cast, print statemetns ok"""
        pass



if __name__ == '__main__':
    IMP.test.main()
