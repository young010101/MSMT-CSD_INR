import unittest
from src import smt_axon_diameter, gmr
import numpy as np
import math as math
import numpy.testing as npt


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(gmr, 2.67e8)

    def test_smt_axon_diameter_origin(self):
        model_param = [0.5, 10e-6, 1.5e-9, 0.2]

        Delta = np.concatenate((np.ones([1, 8]) * 19e-3, np.ones([1, 8]) * 49e-3), axis=1)
        delta = np.concatenate((np.ones([1, 8]) * 8e-3, np.ones([1, 8]) * 8e-3), axis=1)
        bvals = np.array(
            [50, 350, 800, 1500, 2400, 3450, 4750, 6000, 200, 950, 2300, 4250, 6750, 9850, 13500, 17800]) * 1e6
        G = 1. / (gmr * delta) * np.sqrt(bvals / (Delta - delta / 3))
        q = (1 / 2 / math.pi) * gmr * G * delta

        sig = smt_axon_diameter(bvals, Delta, delta, G, model_param)
        self.assertEqual(sig.shape, (1, 16))

        second = np.array([[0.932186, 0.637736, 0.403668, 0.238009, 0.14567 , 0.093919,
        0.059734, 0.040431, 0.772872, 0.384473, 0.198714, 0.121195,
        0.078759, 0.05108 , 0.032739, 0.020327]])
        npt.assert_array_almost_equal(sig, second)


if __name__ == '__main__':
    unittest.main()
