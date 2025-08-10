# Interactive PID Controller Tuning Tool

This project is an interactive web application built with Streamlit for tuning and comparing PID controllers for first and second-order systems. It provides a comprehensive dashboard for performance analysis and visualization.

## 🚀 Features

-   **Interactive System Modeling:** Define first or second-order transfer functions by adjusting parameters like gain, time constant, dead time, natural frequency, and damping ratio.
-   **Multiple Tuning Rules:** Apply industry-standard PID tuning rules to your system model:
    -   Ziegler-Nichols (Open-Loop & Closed-Loop)
    -   Cohen-Coon
    -   IMC (Internal Model Control)
-   **Comprehensive Performance Analysis:** Instantly view key performance metrics for your tuned controller, including:
    -   Time-domain characteristics (Overshoot, Settling Time, Rise Time)
    -   Integral error criteria (IAE, ISE, ITAE, ITSE)
    -   Frequency-domain stability margins (Gain Margin, Phase Margin)
-   **Rich Visualizations:**
    -   Interactive Step Response plot (via Plotly)
    -   Bode plot
    -   Nyquist diagram
-   **Real-Time Interactivity:** All metrics and plots update instantly as you adjust parameters, providing immediate feedback.
-   **Data Export:** Download calculated metrics as a CSV file or export plots as PNG images.

## 🛠️ Tech Stack

-   **Framework:** Streamlit
-   **Backend:** Python
-   **Core Libraries:**
    -   `pandas` for data handling
    -   `numpy` for numerical operations
    -   `python-control` for control system analysis
    -   `plotly` for interactive charts
    -   `matplotlib` for frequency response plots
-   **Frontend Styling:** Tailwind CSS (via CDN)

## 🏃‍♂️ How to Run

1.  **Clone the repository.**

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

4.  Open your web browser and navigate to the local URL provided by Streamlit.
