# 🍷 Deep-Bordeaux: Neural Network Wine Quality Inference

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit)](https://bordeauxwinepredictor.streamlit.app/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](#)

A custom-built Multi-Layer Perceptron (MLP) engineered to predict wine quality scores based on physiochemical telemetry. Developed entirely from scratch using **PyTorch**, this project transforms raw tabular data into a real-time, interactive inference engine.

---

## 🧠 The Architecture

Unlike out-of-the-box regression models, this predictor is built on a custom deep learning architecture designed to capture non-linear chemical relationships.

*   **Input Layer:** 11-dimensional feature vector (Acidity, Sulfur levels, pH, etc.)
*   **Hidden Layers:**
    *   **Layer 1:** 64-neuron linear transformation with ReLU activation to extract primary chemical signatures.
    *   **Layer 2:** 32-neuron linear transformation with ReLU activation for complex feature crossing.
*   **Output Layer:** Single continuous node predicting the final quality score (0-10).

---

## ⚙️ The Training Regimen

The model weights were optimized through rigorous training cycles:

*   **Loss Function:** Mean Squared Error (MSE) to heavily penalize large deviations from human-assigned quality scores.
*   **Optimization Algorithm:** Adam Optimizer ($lr = 0.001$), dynamically adapting the learning rate during backpropagation to prevent exploding gradients and ensure steady convergence.
*   **Data Pipeline:** Inputs strictly normalized via `StandardScaler` ($\mu = 0, \sigma = 1$) to stabilize the gradients during the forward pass. Data streamed in via PyTorch `DataLoader` for efficient batch processing.

---

## 🚀 The Inference Pipeline

The deployed application does not just run a script; it executes a strict machine learning pipeline in real-time:

1.  **State Injection:** The UI captures user-defined chemical features.
2.  **Standardization:** Raw inputs are instantly transformed using the persisted `scaler.pkl` to match the model's training distribution.
3.  **Tensor Conversion:** Scaled NumPy arrays are converted to 32-bit float PyTorch Tensors.
4.  **Forward Pass:** The tensor flows through the `.pth` model state dictionary (running strictly in `model.eval()` mode with gradient tracking disabled via `torch.no_grad()` for maximum performance).

---

## 💻 Run it Locally

Want to inspect the engine yourself? 

**1. Clone the repository:**
```bash
git clone [https://github.com/kunal-garg-1/Bordeaux_wine_predictor.git](https://github.com/kunal-garg-1/Bordeaux_wine_predictor.git)
cd Bordeaux_wine_predictor
```
**2. Install Depedencies**
```bash
pip install -r requirements.txt
```

**3. Launch the inference server**
```bash
streamlit run app.py
