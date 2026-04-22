import streamlit as st
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Set up the Streamlit page configuration
st.set_page_config(page_title="Continuous LTI System Visualiser", layout="wide")

st.title("Continuous LTI System Visualiser")

# --- UI INPUTS ---
st.sidebar.header("System Parameters")
# Input for numerator and denominator coefficients
num_str = st.sidebar.text_input("Numerator coefficients (space-separated)", "1")
den_str = st.sidebar.text_input("Denominator coefficients (space-separated)", "1 3 2")

st.sidebar.header("Input Signal")
# Providing instructions to the user on how to format the signal input
st.sidebar.markdown(
    """
    Define `x(t)` using `t` and `np` functions.
    Examples:
    - Unit Step: `np.where(t >= 0, 1.0, 0.0)`
    - Unit Impulse (approx): `signal.unit_impulse(len(t), idx='mid')` 
    - Ramp: `np.where(t >= 0, t, 0.0)`
    - Sinusoidal: `np.sin(2 * np.pi * t)`
    - Exponential Decay: `np.exp(-t) * (t >= 0)`
    """
)
# Input from user for the signal expression
x_expr = st.sidebar.text_input("Input Signal Expression x(t)", "np.where(t >= 0, 1.0, 0.0)")

if st.sidebar.button("Analyse"):
    try:
        # Parse numerator and denominator coefficients
        num = [float(c) for c in num_str.split()]
        den = [float(c) for c in den_str.split()]
        system = signal.TransferFunction(num, den)
        
        # Time array for the input signal (symmetric to properly check for even/odd symmetry)
        t = np.linspace(-10, 10, 2000)
        dt = t[1] - t[0]
        
        # Safely evaluate the user's input signal expression
        allowed_globals = {'np': np, 't': t, 'signal': signal}
        x = eval(x_expr, {"__builtins__": None}, allowed_globals)
        
        # If user provides a constant, broadcast to the time array
        if np.isscalar(x):
            x = np.ones_like(t) * x
            
        # Ensure the shape matches
        if len(x) != len(t):
            x = np.ones_like(t) * x

        # System Responses setup
        # Impulse and Step responses are causal, computing for t >= 0
        t_resp = np.linspace(0, 20, 2000)
        
        # Compute impulse response h(t) using scipy.signal
        t_resp_sys, h = signal.impulse(system, T=t_resp)
        
        # Compute step response s(t) using scipy.signal
        t_step_sys, s = signal.step(system, T=t_resp)
        
        # Convolution for Output
        # Convolving input signal x(t) with impulse response h(t)
        # Scaling by dt to approximate continuous-time convolution integral
        y_conv = np.convolve(x, h, mode='full') * (t_resp[1] - t_resp[0])
        # Generate the time array for the convolved signal
        t_conv = np.linspace(t[0], t[-1] + t_resp[-1], len(y_conv))
        
        # Interpolate the output signal to map it back onto the original time array for property analysis
        y_interp = np.interp(t, t_conv, y_conv)
        
        # --- Properties Computation Function ---
        def analyze_signal(sig, time_arr):
            """Analyzes numerical properties of a given signal."""
            # Energy: Integral of magnitude squared
            energy = np.sum(np.abs(sig)**2) * dt
            # Power: Average energy over the interval
            power = np.mean(np.abs(sig)**2)
            
            # Classification based on Energy/Power limits
            if np.isfinite(energy) and energy < 1e4 and energy > 1e-4: 
                ep_type = "Energy Signal"
            elif np.isfinite(power) and power > 1e-4:
                ep_type = "Power Signal"
            else:
                ep_type = "Neither"
                
            # Periodicity using autocorrelation
            autocorr = np.correlate(sig, sig, mode='full')
            # Look for repeated peaks
            peaks, _ = signal.find_peaks(autocorr, distance=int(1.0/dt))
            if len(peaks) > 2 and ep_type != "Energy Signal":
                 is_periodic = "Periodic"
            else:
                 is_periodic = "Aperiodic"
            
            # Symmetry check
            if np.allclose(time_arr, -time_arr[::-1]):
                sig_reversed = sig[::-1]
                if np.allclose(sig, sig_reversed, atol=1e-2):
                    sym = "Even"
                elif np.allclose(sig, -sig_reversed, atol=1e-2):
                    sym = "Odd"
                else:
                    sym = "Neither"
            else:
                sym = "N/A"
                
            # Boundedness: Check if the maximum absolute value is below a threshold
            max_val = np.max(np.abs(sig))
            if max_val < 1e5:
                bound = "Bounded"
            else:
                bound = "Unbounded"
                
            return energy, ep_type, is_periodic, sym, bound

        # Compute poles of the system using numpy roots function
        poles = np.roots(den)
        
        # Determine stability from poles
        if np.all(np.real(poles) < -1e-6):
            stability = "BIBO Stable"
            st_class = st.success
        elif np.any(np.real(poles) > 1e-6):
            stability = "Unstable"
            st_class = st.error
        else:
            stability = "Marginally Stable"
            st_class = st.warning
            
        # --- Plotting section (2x2 grid) ---
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Input signal plot
        axs[0, 0].plot(t, x, label='x(t)')
        axs[0, 0].set_title('Input Signal')
        axs[0, 0].set_xlabel('Time (t)')
        axs[0, 0].grid(True)
        axs[0, 0].legend()
        
        # 2. Impulse response plot
        axs[0, 1].plot(t_resp_sys, h, color='orange', label='h(t)')
        axs[0, 1].set_title('Impulse Response')
        axs[0, 1].set_xlabel('Time (t)')
        axs[0, 1].grid(True)
        axs[0, 1].legend()
        
        # 3. Output plot via Convolution
        axs[1, 0].plot(t_conv, y_conv, color='green', label='y(t)')
        axs[1, 0].set_title('y(t) = x(t) * h(t) via Convolution')
        axs[1, 0].set_xlabel('Time (t)')
        axs[1, 0].set_xlim([t[0], t[-1]]) # Focus on interesting time span
        axs[1, 0].grid(True)
        axs[1, 0].legend()
        
        # 4. Step Response plot
        axs[1, 1].plot(t_step_sys, s, color='red', label='Step Response')
        axs[1, 1].set_title('Step Response')
        axs[1, 1].set_xlabel('Time (t)')
        axs[1, 1].grid(True)
        axs[1, 1].legend()
        
        plt.tight_layout()
        st.pyplot(fig) # Display the figure in Streamlit
        
        # --- Properties Panel ---
        st.header("Properties Panel")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Input Signal x(t)")
            e, ep, per, sym, bound = analyze_signal(x, t)
            st.metric("Energy", f"{e:.2e}" if e > 1000 else f"{e:.4f}")
            st.info(f"**Type:** {ep}\n\n**Periodicity:** {per}\n\n**Symmetry:** {sym}\n\n**Boundedness:** {bound}")
            
        with col2:
            st.subheader("System")
            st.markdown("**Poles:**")
            for p in poles:
                st.code(f"{p:.4f}")
            st_class(f"**Stability:** {stability}") # Dynamic styling based on stability
            st.info("**Causality:** Causal (h(t) = 0 for t < 0)")
            
        with col3:
            st.subheader("Output Signal y(t)")
            # Analyze output signal over its interpolated domain
            e_y, ep_y, per_y, _, bound_y = analyze_signal(y_interp, t)
            st.metric("Energy", f"{e_y:.2e}" if e_y > 1000 else f"{e_y:.4f}")
            st.info(f"**Type:** {ep_y}\n\n**Periodicity:** {per_y}\n\n**Boundedness:** {bound_y}")
            
    except Exception as e:
        st.error(f"Error processing input: {e}")
