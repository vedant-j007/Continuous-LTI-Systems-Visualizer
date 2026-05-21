# 🚀 Continuous LTI System Visualiser

An interactive web application for visualising and analysing **Continuous-Time Linear Time-Invariant (LTI) Systems** using Python and Streamlit.

This app allows users to:
- Define custom transfer functions
- Enter custom input signals
- Visualise impulse & step responses
- Compute convolution outputs
- Analyse signal properties and system stability

Built for students, engineers, and signal-processing enthusiasts 📈⚡

---

# ✨ Features

## 📊 System Analysis
- Transfer Function based LTI system modelling
- Pole computation
- Stability detection
- Causal system analysis

## 📈 Signal Visualisation
- Input signal plotting
- Impulse response plotting
- Step response plotting
- Output signal via convolution

## 🧠 Signal Property Analysis
For both input and output signals:
- Energy / Power classification
- Periodicity detection
- Symmetry detection
- Boundedness checking

## ⚡ Interactive UI
- Built entirely using Streamlit
- Real-time signal analysis
- User-defined mathematical expressions

---

# 🖼️ Dashboard Layout

```text
┌──────────────────────┬──────────────────────┐
│     Input Signal     │   Impulse Response   │
├──────────────────────┼──────────────────────┤
│  Convolution Output  │    Step Response     │
└──────────────────────┴──────────────────────┘
```

---

# 🛠️ Tech Stack

- Python
- Streamlit
- NumPy
- SciPy
- Matplotlib

---

# 📂 Project Structure

```bash
.
├── app.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/lti-system-visualiser.git
cd lti-system-visualiser
```

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 3️⃣ Run the App

```bash
streamlit run app.py
```

---

# 📦 Requirements

```txt
streamlit
numpy
scipy
matplotlib
```

---

# 🧪 Example Input Signals

## Unit Step

```python
np.where(t >= 0, 1.0, 0.0)
```

## Ramp Signal

```python
np.where(t >= 0, t, 0.0)
```

## Sinusoidal Signal

```python
np.sin(2 * np.pi * t)
```

## Exponential Decay

```python
np.exp(-t) * (t >= 0)
```

---

# 🧠 Mathematical Background

The output of an LTI system is computed using continuous-time convolution:

\[
y(t) = x(t) * h(t)
\]

\[
y(t)=\int_{-\infty}^{\infty}x(\tau)h(t-\tau)d\tau
\]

where:
- \(x(t)\) → input signal
- \(h(t)\) → impulse response
- \(y(t)\) → output signal

---

# 📊 Stability Analysis

The system poles are computed using:

```python
np.roots(den)
```

The system is classified as:
- ✅ BIBO Stable
- ⚠️ Marginally Stable
- ❌ Unstable

depending on pole locations.

---

# 🎯 Applications

- Signals & Systems learning
- Control Systems analysis
- DSP experimentation
- Engineering education
- Transfer function visualisation

---

# 🚀 Future Improvements

- Pole-Zero Plot
- Bode Plot Visualiser
- Nyquist Plot
- Root Locus
- Fourier Transform Module
- Discrete-Time System Support
- Dark Mode UI
