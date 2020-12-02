"""
Tests determinant functions.
"""

from unittest import TestCase, skipIf
import numpy as np
from numpy.testing import run_module_suite, assert_allclose
from pkg_resources import parse_version
import gulinalg

n_batch = 64


class TestSlogdet(TestCase):
    """sign and (natural) logarithm of the determinant of an array."""

    def test_real(self):
        m = 3
        a = np.random.randn(m, m)
        sign, logdet = gulinalg.slogdet(a)
        det = gulinalg.det(a)
        assert_allclose(det, sign * np.exp(logdet))

    def test_complex(self):
        m = 3
        a = np.random.randn(m, m) + 1j * np.random.randn(m, m)
        sign, logdet = gulinalg.slogdet(a)
        det = gulinalg.det(a)
        assert_allclose(det, sign * np.exp(logdet))

    def test_real_vector(self):
        m = 3
        a = np.random.randn(n_batch, m, m)
        for workers in [1, -1]:
            sign, logdet = gulinalg.slogdet(a, workers=workers)
            det = gulinalg.det(a, workers=workers)
            assert_allclose(det, sign * np.exp(logdet))

    def test_complex_vector(self):
        m = 3
        a = np.random.randn(n_batch, m, m) + 1j*np.random.randn(n_batch, m, m)
        for workers in [1, -1]:
            sign, logdet = gulinalg.slogdet(a, workers=workers)
            det = gulinalg.det(a, workers=workers)
            assert_allclose(det, sign * np.exp(logdet))


if __name__ == '__main__':
    run_module_suite()
