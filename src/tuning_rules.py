"""
PID Tuning Rules Engine

This module contains implementations of various PID controller tuning rules,
including open-loop and closed-loop methods.
"""

import numpy as np
import control as ct

def ziegler_nichols_open_loop(k_p: float, tau: float, theta: float) -> dict:
    """
    Calculates PID parameters using the Ziegler-Nichols open-loop method.
    Based on FOPDT model parameters.

    Args:
        k_p (float): Process Gain (K)
        tau (float): Time Constant (T)
        theta (float): Dead Time (L)

    Returns:
        dict: A dictionary with PID parameters {'Kp', 'Ki', 'Kd'}.
    """
    if k_p == 0:
        raise ValueError("Process gain (k_p) cannot be zero.")
    if theta <= 0:
        raise ValueError("Ziegler-Nichols open-loop method is not applicable for systems with no dead time (theta <= 0).")

    # Classic Z-N formulas for a PID controller
    kc = (1.2 * tau) / (k_p * theta)
    ti = 2.0 * theta
    td = 0.5 * theta

    # Convert to standard PID form (Kp, Ki, Kd)
    kp_controller = kc
    ki_controller = kc / ti
    kd_controller = kc * td

    return {'Kp': kp_controller, 'Ki': ki_controller, 'Kd': kd_controller}


def cohen_coon(k_p: float, tau: float, theta: float) -> dict:
    """
    Calculates PID parameters using the Cohen-Coon method.
    Based on FOPDT model parameters.

    Args:
        k_p (float): Process Gain (K)
        tau (float): Time Constant (T)
        theta (float): Dead Time (L)

    Returns:
        dict: A dictionary with PID parameters {'Kp', 'Ki', 'Kd'}.
    """
    if k_p == 0:
        raise ValueError("Process gain (k_p) cannot be zero.")
    if theta <= 0:
        raise ValueError("Cohen-Coon method is not applicable for systems with no dead time (theta <= 0).")

    L = theta
    T = tau
    K = k_p

    # Cohen-Coon formulas for a PID controller
    kc = (T / (K * L)) * (4/3 + (L / (4 * T)))
    ti = L * ( (32 * T + 6 * L) / (13 * T + 8 * L) )
    td = L * ( (4 * T) / (11 * T + 2 * L) )

    # Convert to standard PID form (Kp, Ki, Kd)
    kp_controller = kc
    ki_controller = kc / ti
    kd_controller = kc * td

    return {'Kp': kp_controller, 'Ki': ki_controller, 'Kd': kd_controller}


def ziegler_nichols_closed_loop(tf_model: ct.TransferFunction) -> dict:
    """
    Calculates PID parameters using the Ziegler-Nichols closed-loop method.
    This method requires finding the ultimate gain (Ku) and ultimate period (Tu).

    Args:
        tf_model (control.TransferFunction): The transfer function of the system.

    Returns:
        dict: A dictionary with PID parameters {'Kp', 'Ki', 'Kd'}.
    """
    gm, _, wg, _ = ct.margin(tf_model)

    if np.isinf(gm) or gm <= 0:
        raise ValueError("System does not have a finite ultimate gain (Ku). The Ziegler-Nichols closed-loop method is not applicable.")

    ku = gm
    # The frequency wg is in rad/s. Tu is the period.
    if wg <= 0:
        raise ValueError("System does not have a finite ultimate period (Tu). The Ziegler-Nichols closed-loop method is not applicable.")
    tu = 2 * np.pi / wg

    # Classic Z-N formulas for a PID controller
    kc = 0.6 * ku
    ti = tu / 2.0
    td = tu / 8.0

    # Convert to standard PID form (Kp, Ki, Kd)
    kp_controller = kc
    ki_controller = kc / ti
    kd_controller = kc * td

    return {'Kp': kp_controller, 'Ki': ki_controller, 'Kd': kd_controller}


def imc(k_p: float, tau: float, theta: float, tau_c: float) -> dict:
    """
    Calculates PID parameters using the IMC (Internal Model Control) method.
    Based on FOPDT model parameters and a desired closed-loop time constant.

    Args:
        k_p (float): Process Gain (K)
        tau (float): Time Constant (T)
        theta (float): Dead Time (L)
        tau_c (float): Desired closed-loop time constant (lambda).

    Returns:
        dict: A dictionary with PID parameters {'Kp', 'Ki', 'Kd'}.
    """
    if k_p == 0:
        raise ValueError("Process gain (k_p) cannot be zero.")
    if tau_c <= 0:
        raise ValueError("Closed-loop time constant (tau_c) must be positive.")

    K = k_p
    T = tau
    L = theta

    # IMC-based PID tuning rules for a PID controller
    kc = (1 / K) * (T + 0.5 * L) / (tau_c + 0.5 * L)
    ti = T + 0.5 * L
    td = (T * L) / (2 * T + L) if (2 * T + L) != 0 else 0

    # Convert to standard PID form (Kp, Ki, Kd)
    kp_controller = kc
    ki_controller = kc / ti if ti != 0 else 0
    kd_controller = kc * td

    return {'Kp': kp_controller, 'Ki': ki_controller, 'Kd': kd_controller}
