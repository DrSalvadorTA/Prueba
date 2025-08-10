import control as ct
import numpy as np

class FirstOrderTransferFunction:
    """
    Represents a first-order plus dead time (FOPDT) transfer function.
    G(s) = Kp / (τs + 1) * e^(-θs)
    """
    def __init__(self, Kp: float, tau: float, theta: float):
        """
        Initializes the first-order transfer function.

        Args:
            Kp (float): Process gain.
            tau (float): Time constant.
            theta (float): Dead time.
        """
        if tau <= 0:
            raise ValueError("Time constant (tau) must be positive.")
        if theta < 0:
            raise ValueError("Dead time (theta) cannot be negative.")

        self.Kp = Kp
        self.tau = tau
        self.theta = theta

        # Use Pade approximation for dead time
        if self.theta > 0:
            num, den = ct.pade(self.theta, 1)
            dead_time_tf = ct.tf(num, den)
        else:
            dead_time_tf = ct.tf([1], [1])

        self.tf_model = ct.tf([self.Kp], [self.tau, 1]) * dead_time_tf

    def __repr__(self):
        return f"FirstOrderTransferFunction(Kp={self.Kp}, tau={self.tau}, theta={self.theta})"

class SecondOrderTransferFunction:
    """
    Represents a second-order plus dead time transfer function.
    G(s) = Kp * ωn² / (s² + 2ζωns + ωn²) * e^(-θs)
    """
    def __init__(self, Kp: float, omega_n: float, zeta: float, theta: float):
        """
        Initializes the second-order transfer function.

        Args:
            Kp (float): Process gain.
            omega_n (float): Natural frequency.
            zeta (float): Damping ratio.
            theta (float): Dead time.
        """
        if omega_n <= 0:
            raise ValueError("Natural frequency (omega_n) must be positive.")
        if zeta <= 0:
            raise ValueError("Damping ratio (zeta) must be positive for a stable system.")
        if theta < 0:
            raise ValueError("Dead time (theta) cannot be negative.")

        self.Kp = Kp
        self.omega_n = omega_n
        self.zeta = zeta
        self.theta = theta

        # Use Pade approximation for dead time
        if self.theta > 0:
            num, den = ct.pade(self.theta, 1)
            dead_time_tf = ct.tf(num, den)
        else:
            dead_time_tf = ct.tf([1], [1])

        numerator = [self.Kp * self.omega_n**2]
        denominator = [1, 2 * self.zeta * self.omega_n, self.omega_n**2]

        self.tf_model = ct.tf(numerator, denominator) * dead_time_tf

    def __repr__(self):
        return f"SecondOrderTransferFunction(Kp={self.Kp}, omega_n={self.omega_n}, zeta={self.zeta}, theta={self.theta})"
