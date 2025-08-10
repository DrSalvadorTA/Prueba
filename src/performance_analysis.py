"""
Performance Analysis Engine

This module contains functions to simulate PID controlled systems
and calculate various performance metrics in both the time and frequency domains.
"""
import numpy as np
import control as ct

def get_pid_controller(pid_params: dict) -> ct.TransferFunction:
    """Creates a PID controller transfer function from gains."""
    kp = pid_params.get('Kp', 0)
    ki = pid_params.get('Ki', 0)
    kd = pid_params.get('Kd', 0)

    # C(s) = Kp + Ki/s + Kd*s = (Kd*s^2 + Kp*s + Ki) / s
    # This assumes the denominator is 's'. If ki is 0, this is not ideal.
    # A more robust way is to build it from components.
    p = ct.tf([kp], [1])
    i = ct.tf([ki], [1, 0]) if ki != 0 else 0
    d = ct.tf([kd, 0], [1]) if kd != 0 else 0 # This is Kd*s

    controller_tf = p + i + d
    # Handle the case where all gains are zero
    if not isinstance(controller_tf, ct.TransferFunction):
        return ct.tf([0], [1])
    return controller_tf

def analyze_step_response(sys_tf: ct.TransferFunction, pid_params: dict) -> dict:
    """
    Simulates the closed-loop step response and calculates performance characteristics.

    Args:
        sys_tf (ct.TransferFunction): The transfer function of the system.
        pid_params (dict): A dictionary with PID parameters {'Kp', 'Ki', 'Kd'}.

    Returns:
        dict: A dictionary containing 'time', 'output', and 'info' (from step_info).
    """
    controller_tf = get_pid_controller(pid_params)

    if np.all(controller_tf.num[0][0] == [0]): # Check if controller is zero
        return {
            "time": np.array([0, 1]),
            "output": np.array([0, 0]),
            "info": {'RiseTime': np.nan, 'SettlingTime': np.nan, 'Overshoot': np.nan, 'Peak': np.nan, 'PeakTime': np.nan}
        }

    # Form the closed-loop system
    closed_loop_tf = ct.feedback(controller_tf * sys_tf, 1)

    # Determine a suitable simulation time
    try:
        poles = ct.poles(closed_loop_tf)
        if len(poles) == 0:
            T_final = 50
        else:
            # Look at the real parts of the poles to find the slowest dynamic
            real_parts = sorted([abs(p.real) for p in poles if p.real < -1e-6])
            if not real_parts: # Pure integrator or unstable
                 T_final = 50
            else:
                 slowest_pole_decay = real_parts[0]
                 T_final = 10 / slowest_pole_decay # 10 time constants
    except Exception:
        T_final = 50 # Default simulation time as a fallback

    time_vec = np.linspace(0, T_final, 1500)
    t, y = ct.step_response(closed_loop_tf, T=time_vec)

    # Get step info
    info = ct.step_info(closed_loop_tf, T=time_vec, SettlingTimeThreshold=0.02)

    return {"time": t, "output": y, "info": info}


def calculate_performance_indices(time: np.ndarray, output: np.ndarray, setpoint: float = 1.0) -> dict:
    """
    Calculates integral performance indices from step response data.

    Args:
        time (np.ndarray): The time vector from the simulation.
        output (np.ndarray): The output vector from the simulation.
        setpoint (float): The setpoint of the step response. Defaults to 1.0.

    Returns:
        dict: A dictionary with the calculated indices {IAE, ISE, ITAE, ITSE}.
    """
    if len(time) != len(output) or len(time) < 2:
        return {'IAE': np.nan, 'ISE': np.nan, 'ITAE': np.nan, 'ITSE': np.nan}

    error = setpoint - output

    # Integral Absolute Error
    iae = np.trapezoid(np.abs(error), time)

    # Integral Square Error
    ise = np.trapezoid(error**2, time)

    # Integral Time Absolute Error
    itae = np.trapezoid(time * np.abs(error), time)

    # Integral Time Square Error
    itse = np.trapezoid(time * error**2, time)

    return {'IAE': iae, 'ISE': ise, 'ITAE': itae, 'ITSE': itse}


def analyze_frequency_domain(sys_tf: ct.TransferFunction, pid_params: dict) -> dict:
    """
    Calculates frequency domain stability margins for the controlled system.

    Args:
        sys_tf (ct.TransferFunction): The transfer function of the system.
        pid_params (dict): A dictionary with PID parameters {'Kp', 'Ki', 'Kd'}.

    Returns:
        dict: A dictionary with the calculated margins {GainMargin, PhaseMargin}.
    """
    controller_tf = get_pid_controller(pid_params)

    # The margins are calculated on the open-loop system: controller * plant
    open_loop_tf = controller_tf * sys_tf

    # margin() returns gm in linear scale, pm in degrees
    gm, pm, _, _ = ct.margin(open_loop_tf)

    # Handle infinite margins for stable systems
    gain_margin_val = gm if not np.isinf(gm) else 'stable'
    phase_margin_val = pm if not np.isinf(pm) else 'stable'

    return {'GainMargin': gain_margin_val, 'PhaseMargin': phase_margin_val}
