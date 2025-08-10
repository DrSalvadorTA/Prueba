import unittest
import numpy as np
import control as ct
from src.performance_analysis import (
    analyze_step_response,
    calculate_performance_indices,
    analyze_frequency_domain,
)
from src.transfer_functions import FirstOrderTransferFunction

class TestPerformanceAnalysis(unittest.TestCase):

    def test_step_response_analysis(self):
        """Test the step response analysis for a known system."""
        # G(s) = 1 / (s^2 + s + 1), with P-controller Kp=1
        # Closed-loop: 1 / (s^2 + s + 2)
        sys_tf = ct.tf([1], [1, 1, 1])
        pid_params = {'Kp': 1, 'Ki': 0, 'Kd': 0}

        result = analyze_step_response(sys_tf, pid_params)
        info = result['info']

        self.assertAlmostEqual(info['Overshoot'], 30.5, delta=1.5)
        self.assertAlmostEqual(info['SettlingTime'], 7.5, delta=0.5)
        self.assertAlmostEqual(info['RiseTime'], 1.0, delta=0.2)

    def test_performance_indices(self):
        """Test the performance indices calculation with an analytical case."""
        # Linear response from (0,0) to (2,1)
        time = np.linspace(0, 2, 100)
        output = 0.5 * time

        indices = calculate_performance_indices(time, output)

        # Analytical values: IAE=1, ISE=2/3, ITAE=2/3, ITSE=1/3
        self.assertAlmostEqual(indices['IAE'], 1.0, places=3)
        self.assertAlmostEqual(indices['ISE'], 0.667, places=3)
        self.assertAlmostEqual(indices['ITAE'], 0.667, places=3)
        self.assertAlmostEqual(indices['ITSE'], 0.333, places=3)

    def test_frequency_analysis(self):
        """Test the frequency domain analysis for a known system."""
        # G(s) = 1 / ((s+1)^3), with P-controller Kp=4
        sys_tf = ct.tf([1], [1, 3, 3, 1])
        pid_params = {'Kp': 4, 'Ki': 0, 'Kd': 0}

        margins = analyze_frequency_domain(sys_tf, pid_params)

        self.assertAlmostEqual(margins['GainMargin'], 2.0, delta=0.01)
        self.assertAlmostEqual(margins['PhaseMargin'], 27.1, delta=0.2)

if __name__ == '__main__':
    unittest.main()
