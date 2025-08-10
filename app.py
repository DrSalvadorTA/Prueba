import streamlit as st
import pandas as pd
import io
from src.transfer_functions import FirstOrderTransferFunction, SecondOrderTransferFunction
from src.tuning_rules import (
    ziegler_nichols_open_loop,
    cohen_coon,
    imc,
    ziegler_nichols_closed_loop,
)
from src.performance_analysis import (
    analyze_step_response,
    calculate_performance_indices,
    analyze_frequency_domain,
    get_pid_controller,
)
from src.plotting import plot_step_response, plot_bode, plot_nyquist

# Set page config
st.set_page_config(layout="wide", page_title="PID Controller Tuning")

# --- Custom Styling (Tailwind CSS) ---
st.html("""
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .stApp { background-color: #1a202c; color: #e2e8f0; }
        .st-emotion-cache-16txtl3 { padding-top: 2rem; }
        .stSidebar { background-color: #2d3748; }
        h1, h2, h3, h4, h5, h6 { color: #f7fafc; }
    </style>
""")

# --- Main Title ---
st.html('<h1 class="text-4xl font-bold text-center mb-8">PID Controller Tuning Comparison</h1>')

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.html('<h2 class="text-2xl font-semibold mb-4">System Parameters</h2>')
    model_type = st.selectbox("Select Transfer Function Model", ("First-Order", "Second-Order"))
    st.markdown("---")

    if model_type == "First-Order":
        st.html('<h3 class="text-xl font-medium mb-2">First-Order Model</h3>')
        kp_sys = st.number_input("Process Gain (Kp)", -10.0, 10.0, 1.0, 0.1, key="fopdt_kp")
        tau = st.slider("Time Constant (τ)", 0.1, 20.0, 5.0, 0.1, key="fopdt_tau")
        theta = st.slider("Dead Time (θ)", 0.0, 10.0, 1.0, 0.1, key="fopdt_theta")
        tuning_methods = ["Ziegler-Nichols (Open Loop)", "Cohen-Coon", "IMC"]
    else: # Second-Order
        st.html('<h3 class="text-xl font-medium mb-2">Second-Order Model</h3>')
        kp_sys = st.number_input("Process Gain (Kp)", -10.0, 10.0, 1.0, 0.1, key="sopdt_kp")
        omega_n = st.slider("Natural Frequency (ωn)", 0.1, 10.0, 1.0, 0.1, key="sopdt_omega")
        zeta = st.slider("Damping Ratio (ζ)", 0.1, 2.0, 0.5, 0.05, key="sopdt_zeta")
        theta = st.slider("Dead Time (θ)", 0.0, 10.0, 1.0, 0.1, key="sopdt_theta")
        tuning_methods = ["Ziegler-Nichols (Closed Loop)"]

    st.markdown("---")
    st.html('<h2 class="text-2xl font-semibold mb-4">Tuning Rule</h2>')
    selected_method = st.selectbox("Select Tuning Method", tuning_methods)

    tau_c = None
    if selected_method == "IMC":
        st.html('<h3 class="text-xl font-medium mb-2">IMC Parameter</h3>')
        tau_c = st.slider("Closed-loop time constant (τc)", 0.1, 10.0, 1.0, 0.1)

    # --- Help Section ---
    st.markdown("---")
    with st.expander("About this App & Help", expanded=False):
        st.markdown("""
            **What is this?**

            This application is an interactive tool for tuning PID controllers. You can define a system, choose a tuning rule, and instantly see the controller's performance through various metrics and plots.

            ---

            **Tuning Rules Explained**

            *   **Ziegler-Nichols (Open Loop):** A classic heuristic method based on the system's reaction to a step input. Works well for systems with a clear "S"-shaped response.
            *   **Cohen-Coon:** An extension of the Z-N method, also based on the step response. It often provides more aggressive control.
            *   **IMC (Internal Model Control):** A more model-based approach where you tune the controller based on a desired closed-loop response time (`τc`). Smaller `τc` gives a faster response, but can reduce stability.
            *   **Ziegler-Nichols (Closed Loop):** A method based on finding the point where the system becomes marginally stable under proportional control. It uses the "ultimate gain" and "ultimate period" of the system.

            ---

            **Key Metrics Explained**

            *   **Overshoot:** How much the response exceeds the final setpoint value.
            *   **Settling Time:** The time it takes for the response to stay within a certain percentage (here, 2%) of the final value.
            *   **Rise Time:** The time it takes to go from 10% to 90% of the final value.
            *   **IAE, ISE, etc.:** Integral error criteria that quantify the total error over time. Lower values are generally better.
            *   **Gain/Phase Margin:** Measures of the system's stability. Higher margins indicate a more stable system.
        """)

# --- Main Content Area ---
try:
    # --- 1. Get System TF and Calculate PID Params ---
    if model_type == "First-Order":
        system = FirstOrderTransferFunction(kp_sys, tau, theta)
        if selected_method == "Ziegler-Nichols (Open Loop)":
            pid_params = ziegler_nichols_open_loop(kp_sys, tau, theta)
        elif selected_method == "Cohen-Coon":
            pid_params = cohen_coon(kp_sys, tau, theta)
        else: # IMC
            pid_params = imc(kp_sys, tau, theta, tau_c)
    else: # Second-Order
        system = SecondOrderTransferFunction(kp_sys, omega_n, zeta, theta)
        pid_params = ziegler_nichols_closed_loop(system.tf_model)

    # --- 2. Run Performance Analysis ---
    step_response_data = analyze_step_response(system.tf_model, pid_params)
    indices = calculate_performance_indices(step_response_data['time'], step_response_data['output'])
    margins = analyze_frequency_domain(system.tf_model, pid_params)
    controller_tf = get_pid_controller(pid_params)
    open_loop_tf = controller_tf * system.tf_model

    # --- 3. Display Results ---
    st.html('<h2 class="text-3xl font-semibold mb-6">Performance Analysis</h2>')

    col1, col2 = st.columns(2)
    with col1:
        st.html('<h3 class="text-xl font-medium mb-2">PID Gains</h3>')
        st.html(f"""
            <div class="grid grid-cols-3 gap-2 text-center">
                <div class="bg-gray-700 p-3 rounded-lg"><p class="text-base text-gray-400">Kp</p><p class="text-2xl">{pid_params['Kp']:.3f}</p></div>
                <div class="bg-gray-700 p-3 rounded-lg"><p class="text-base text-gray-400">Ki</p><p class="text-2xl">{pid_params['Ki']:.3f}</p></div>
                <div class="bg-gray-700 p-3 rounded-lg"><p class="text-base text-gray-400">Kd</p><p class="text-2xl">{pid_params['Kd']:.3f}</p></div>
            </div>
        """)

    with col2:
        st.html('<h3 class="text-xl font-medium mb-2">Stability Margins</h3>')
        gm_val = f"{margins['GainMargin']:.2f}" if isinstance(margins['GainMargin'], float) else margins['GainMargin']
        pm_val = f"{margins['PhaseMargin']:.2f}°" if isinstance(margins['PhaseMargin'], float) else margins['PhaseMargin']
        st.html(f"""
            <div class="grid grid-cols-2 gap-2 text-center">
                <div class="bg-gray-700 p-3 rounded-lg"><p class="text-base text-gray-400">Gain Margin</p><p class="text-2xl">{gm_val}</p></div>
                <div class="bg-gray-700 p-3 rounded-lg"><p class="text-base text-gray-400">Phase Margin</p><p class="text-2xl">{pm_val}</p></div>
            </div>
        """)

    st.markdown("<hr class='my-6 border-gray-700'>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.html('<h3 class="text-xl font-medium mb-2">Step Response Info</h3>')
        step_info_clean = step_response_data['info']
        step_info_clean.pop('SteadyStateValue', None)
        step_info_df = pd.DataFrame([step_info_clean]).T
        step_info_df.columns = ["Value"]
        st.dataframe(step_info_df.style.format("{:.2f}"))

    with col4:
        st.html('<h3 class="text-xl font-medium mb-2">Performance Indices</h3>')
        indices_df = pd.DataFrame([indices]).T
        indices_df.columns = ["Value"]
        st.dataframe(indices_df.style.format("{:.2f}"))

    st.markdown("<hr class='my-8 border-gray-700'>", unsafe_allow_html=True)
    st.html('<h2 class="text-3xl font-semibold mb-6">Visualizations</h2>')

    # --- 4. Display Plots in Tabs ---
    tab1, tab2, tab3 = st.tabs(["Step Response", "Bode Plot", "Nyquist Plot"])

    with tab1:
        step_fig = plot_step_response(step_response_data['time'], step_response_data['output'])
        st.plotly_chart(step_fig, use_container_width=True, config={'toImageButtonOptions': {'format': 'png', 'filename': 'step_response'}})

    with tab2:
        bode_fig = plot_bode(open_loop_tf)
        st.pyplot(bode_fig, use_container_width=True)
        buf = io.BytesIO()
        bode_fig.savefig(buf, format="png", bbox_inches="tight", facecolor=bode_fig.get_facecolor())
        st.download_button("Download Bode Plot", data=buf, file_name="bode_plot.png", mime="image/png")

    with tab3:
        nyquist_fig = plot_nyquist(open_loop_tf)
        st.pyplot(nyquist_fig, use_container_width=True)
        buf = io.BytesIO()
        nyquist_fig.savefig(buf, format="png", bbox_inches="tight", facecolor=nyquist_fig.get_facecolor())
        st.download_button("Download Nyquist Plot", data=buf, file_name="nyquist_plot.png", mime="image/png")

    # --- 5. Export Data ---
    with st.sidebar:
        st.markdown("---")
        st.html('<h2 class="text-2xl font-semibold mb-4">Export</h2>')

        # Combine all data into one DataFrame
        full_results = {**pid_params, **margins, **step_info_clean, **indices}
        results_df = pd.DataFrame([full_results]).T
        results_df.columns = ["Value"]

        st.download_button(
            label="Download Results as CSV",
            data=results_df.to_csv().encode('utf-8'),
            file_name=f"pid_analysis_{selected_method.replace(' ', '_').lower()}.csv",
            mime='text/csv',
            use_container_width=True
        )

except (ValueError, np.linalg.LinAlgError, OverflowError) as e:
    st.error(f"""
        **Calculation Error:** {e}

        This can happen when a tuning rule is not applicable or the system is unstable with the current parameters.
        Please check the rule's requirements or adjust the system parameters.
    """)
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
