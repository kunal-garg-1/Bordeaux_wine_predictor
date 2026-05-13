import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib

# 1. Redefining the exact same model architecture so PyTorch knows how to load the weights
class WinePredictor(nn.Module):
    def __init__(self, input_size):
        super(WinePredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        out = self.relu1(self.layer1(x))
        out = self.relu2(self.layer2(out))
        out = self.output_layer(out)
        return out

# 2. Load the model and scaler efficiently (cache it so it doesn't reload on every click)
@st.cache_resource
def load_assets():
    scaler = joblib.load('wine_scaler.pkl')
    model = WinePredictor(input_size=11)
    model.load_state_dict(torch.load('wine_model.pth'))
    model.eval() # Set to evaluation mode
    return model, scaler

model, scaler = load_assets()

# 3. Build the User Interface
st.title("🍷 Bordeaux Wine Quality Predictor")
st.write("Adjust the chemical properties below to predict the quality score (0-10) of the wine.")

# Create columns for a cleaner layout
col1, col2, col3 = st.columns(3)

with col1:
    fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 8.3)
    residual_sugar = st.slider("Residual Sugar", 0.9, 15.5, 2.5)
    total_sulfur = st.slider("Total Sulfur Dioxide", 6.0, 289.0, 46.0)
    sulphates = st.slider("Sulphates", 0.3, 2.0, 0.65)

with col2:
    volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, 0.5)
    chlorides = st.slider("Chlorides", 0.01, 0.61, 0.08)
    density = st.slider("Density", 0.990, 1.003, 0.996)
    alcohol = st.slider("Alcohol (%)", 8.0, 15.0, 10.4)

with col3:
    citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.27)
    free_sulfur = st.slider("Free Sulfur Dioxide", 1.0, 72.0, 14.0)
    pH = st.slider("pH", 2.7, 4.0, 3.3)

# 4. The Prediction Logic
if st.button("Predict Quality"):
    # Gather all inputs in the exact order the model was trained on
    user_input = np.array([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
        chlorides, free_sulfur, total_sulfur, density, pH, sulphates, alcohol
    ]])
    
    # Scale the user's input using the saved scaler
    scaled_input = scaler.transform(user_input)
    
    # Convert to PyTorch Tensor
    tensor_input = torch.tensor(scaled_input, dtype=torch.float32)
    
    # Make the prediction
    with torch.no_grad():
        prediction = model(tensor_input)
        
    # Extract the number and round it
    final_score = prediction.item()
    
    st.success(f"### Predicted Wine Quality: {final_score:.1f} / 10")