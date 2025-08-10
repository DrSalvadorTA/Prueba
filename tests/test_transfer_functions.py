import unittest
import numpy as np
from src.transfer_functions import FirstOrderTransferFunction, SecondOrderTransferFunction
import control as ct

class TestTransferFunctions(unittest.TestCase):

    def test_first_order_creation(self):
        """Test creation of a first-order transfer function."""
        tf = FirstOrderTransferFunction(Kp=2.0, tau=3.0, theta=1.0)
        self.assertEqual(tf.Kp, 2.0)
        self.assertEqual(tf.tau, 3.0)
        self.assertEqual(tf.theta, 1.0)
        # Check if the underlying control model is created
        self.assertIsInstance(tf.tf_model, ct.TransferFunction)

    def test_first_order_invalid_tau(self):
        """Test that a negative tau raises a ValueError."""
        with self.assertRaises(ValueError):
            FirstOrderTransferFunction(Kp=2.0, tau=-3.0, theta=1.0)

    def test_first_order_invalid_theta(self):
        """Test that a negative theta raises a ValueError."""
        with self.assertRaises(ValueError):
            FirstOrderTransferFunction(Kp=2.0, tau=3.0, theta=-1.0)

    def test_second_order_creation(self):
        """Test creation of a second-order transfer function."""
        tf = SecondOrderTransferFunction(Kp=1.0, omega_n=5.0, zeta=0.5, theta=0.5)
        self.assertEqual(tf.Kp, 1.0)
        self.assertEqual(tf.omega_n, 5.0)
        self.assertEqual(tf.zeta, 0.5)
        self.assertEqual(tf.theta, 0.5)
        self.assertIsInstance(tf.tf_model, ct.TransferFunction)

    def test_second_order_invalid_omega_n(self):
        """Test that a non-positive omega_n raises a ValueError."""
        with self.assertRaises(ValueError):
            SecondOrderTransferFunction(Kp=1.0, omega_n=0, zeta=0.5, theta=0.5)
        with self.assertRaises(ValueError):
            SecondOrderTransferFunction(Kp=1.0, omega_n=-5.0, zeta=0.5, theta=0.5)

    def test_second_order_invalid_zeta(self):
        """Test that a non-positive zeta raises a ValueError."""
        with self.assertRaises(ValueError):
            SecondOrderTransferFunction(Kp=1.0, omega_n=5.0, zeta=0, theta=0.5)
        with self.assertRaises(ValueError):
            SecondOrderTransferFunction(Kp=1.0, omega_n=5.0, zeta=-0.5, theta=0.5)

    def test_second_order_invalid_theta(self):
        """Test that a negative theta raises a ValueError."""
        with self.assertRaises(ValueError):
            SecondOrderTransferFunction(Kp=1.0, omega_n=5.0, zeta=0.5, theta=-0.5)

if __name__ == '__main__':
    unittest.main()
