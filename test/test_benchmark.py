from __future__ import print_function
import IMP.test
import IMP.algebra

try:
    import IMP.pynet.benchmark as bench
except ModuleNotFoundError:
    import pyext.src.benchmark as bench
import io
import os
import math
import numpy as np
import jax
import jax.numpy as jnp


class TestBenchmark(IMP.test.TestCase):
    def test_accuracy(self):
        y_ref = np.array([0, 1, 0])
        self.assertEqual(1, bench.accuracy(y_ref, y_ref))

        y_ref = jnp.array(y_ref)
        self.assertEqual(1, bench.accuracy(y_ref, y_ref))

        y_ref = jnp.array([0, 1, 0], dtype=float)
        self.assertEqual(1, bench.accuracy(y_ref, y_ref))

    def test_benchmark(self):
        key = jax.random.PRNGKey(33)
        ground_truth = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])

        nsamples = 1000
        shape = (3, 3, nsamples)
        edge_prob = 2 / 9

        samples = jax.random.bernoulli(key, p=edge_prob, shape=shape)
        self.assertEqual(samples.shape[0:2], ground_truth.shape)

        """Write the test cast, print statemetns ok"""


if __name__ == "__main__":
    IMP.test.main()
