"""
Plotting Engine

This module contains functions for creating visualizations
for the PID controller tuning application using Plotly.
"""
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import control as ct

def plot_step_response(time: np.ndarray, output: np.ndarray, setpoint: float = 1.0) -> go.Figure:
    """
    Creates an interactive Plotly chart of the step response.

    Args:
        time (np.ndarray): The time vector from the simulation.
        output (np.ndarray): The output vector from the simulation.
        setpoint (float): The setpoint of the step response.

    Returns:
        go.Figure: The Plotly figure object.
    """
    fig = go.Figure()

    # Add the step response trace
    fig.add_trace(go.Scatter(
        x=time,
        y=output,
        mode='lines',
        name='System Response',
        line=dict(color='#38bdf8', width=2) # sky-400
    ))

    # Add the setpoint line
    fig.add_trace(go.Scatter(
        x=[time[0], time[-1]] if len(time) > 0 else [0, 1],
        y=[setpoint, setpoint],
        mode='lines',
        name='Setpoint',
        line=dict(color='#f87171', width=2, dash='dash') # red-400
    ))

    fig.update_layout(
        title_text="Closed-Loop Step Response",
        xaxis_title_text="Time (seconds)",
        yaxis_title_text="Output",
        template="plotly_dark",
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top'),
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="#1a202c", # gray-900
        plot_bgcolor="#2d3748", # gray-800
    )

    return fig


def plot_bode(open_loop_tf: ct.TransferFunction) -> plt.Figure:
    """
    Creates a Bode plot for the given open-loop transfer function.

    Args:
        open_loop_tf (ct.TransferFunction): The open-loop system (controller * plant).

    Returns:
        plt.Figure: The Matplotlib figure object containing the Bode plot.
    """
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(8, 6), facecolor="#1a202c")

    # Use the control library's Bode plot function
    # Turn off plotting to the screen, we just want the data to plot ourselves
    mag, phase, omega = ct.bode(open_loop_tf, plot=False)

    # Manually create the plot for better styling control
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.semilogx(omega, 20 * np.log10(mag), color='#38bdf8')
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='#4a5568')
    ax1.set_facecolor("#2d3748")

    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    ax2.semilogx(omega, np.rad2deg(phase), color='#38bdf8')
    ax2.set_ylabel("Phase (degrees)")
    ax2.set_xlabel("Frequency (rad/s)")
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5, color='#4a5568')
    ax2.set_facecolor("#2d3748")

    plt.suptitle("Bode Plot", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_nyquist(open_loop_tf: ct.TransferFunction) -> plt.Figure:
    """
    Creates a Nyquist plot for the given open-loop transfer function.

    Args:
        open_loop_tf (ct.TransferFunction): The open-loop system (controller * plant).

    Returns:
        plt.Figure: The Matplotlib figure object containing the Nyquist plot.
    """
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(7, 7), facecolor="#1a202c")
    ax = fig.add_subplot(1, 1, 1)

    # Use the control library's Nyquist plot function
    ct.nyquist_plot(open_loop_tf, plot=True, ax=ax)

    # Customize the plot
    ax.set_title("Nyquist Plot", fontsize=16)
    ax.set_facecolor("#2d3748")
    ax.grid(True, linestyle='--', linewidth=0.5, color='#4a5568')
    ax.set_xlabel("Real Axis")
    ax.set_ylabel("Imaginary Axis")
    ax.axhline(0, color='#cbd5e0', lw=0.5)
    ax.axvline(0, color='#cbd5e0', lw=0.5)

    fig.tight_layout()
    return fig
