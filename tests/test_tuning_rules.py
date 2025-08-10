import unittest
import numpy as np
import control as ct
from src.tuning_rules import (
    ziegler_nichols_open_loop,
    cohen_coon,
    imc,
    ziegler_nichols_closed_loop,
)

class TestTuningRules(unittest.TestCase):

    def test_zn_open_loop(self):
        """Test the Ziegler-Nichols open-loop tuning rule."""
        params = ziegler_nichols_open_loop(k_p=1.5, tau=4.0, theta=1.0)
        self.assertAlmostEqual(params['Kp'], 3.2, places=3)
        self.assertAlmostEqual(params['Ki'], 1.6, places=3)
        self.assertAlmostEqual(params['Kd'], 1.6, places=3)

    def test_zn_open_loop_no_dead_time(self):
        """Test that Z-N open-loop raises an error for theta=0."""
        with self.assertRaises(ValueError):
            ziegler_nichols_open_loop(k_p=1.5, tau=4.0, theta=0)

    def test_cohen_coon(self):
        """Test the Cohen-Coon tuning rule."""
        params = cohen_coon(k_p=1.5, tau=4.0, theta=1.0)
        self.assertAlmostEqual(params['Kp'], 3.722, places=3)
        self.assertAlmostEqual(params['Ki'], 1.667, places=3)
        self.assertAlmostEqual(params['Kd'], 1.295, places=3)

    def test_imc(self):
        """Test the IMC tuning rule."""
        params = imc(k_p=1.5, tau=4.0, theta=1.0, tau_c=1.5)
        self.assertAlmostEqual(params['Kp'], 1.5, places=3)
        self.assertAlmostEqual(params['Ki'], 0.333, places=3)
        self.assertAlmostEqual(params['Kd'], 0.667, places=3)

    def test_zn_closed_loop(self):
        """Test the Ziegler-Nichols closed-loop tuning rule."""
        # System: G(s) = 1 / ((s+1)^3)
        # Expected: Ku = 8, Tu = 2*pi/sqrt(3) ~= 3.628
        tf_model = ct.tf([1], [1, 3, 3, 1])
        params = ziegler_nichols_closed_loop(tf_model)
        self.assertAlmostEqual(params['Kp'], 4.8, places=3)
        self.assertAlmostEqual(params['Ki'], 2.646, places=3)
        self.assertAlmostEqual(params['Kd'], 2.177, places=3)

    def test_zn_closed_loop_no_crossing(self):
        """Test that Z-N closed-loop raises an error for a first-order system."""
        tf_model = ct.tf([1], [1, 1])
        with self.assertRaises(ValueError):
            ziegler_nichols_closed_loop(tf_model)

if __name__ == '__main__':
    unittest.main()
