import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Title and description
st.title("Diamond Price Predictor")
st.write("Enter the diamond's characteristics to predict its price using a Random Forest model.")

# Load the model
try:
    with open('rf_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: 'rf_model.pkl' not found. Please ensure the file is in the same directory as this script.")
    st.stop()

# Define categorical mappings (same as used during training)
cut_options = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_options = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
clarity_options = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

# Initialize LabelEncoders with the same mappings
le_cut = LabelEncoder()
le_cut.classes_ = np.array(cut_options)
le_color = LabelEncoder()
le_color.classes_ = np.array(color_options)
le_clarity = LabelEncoder()
le_clarity.classes_ = np.array(clarity_options)

# Create input widgets
st.header("Input Diamond Features")
carat = st.slider("Carat", min_value=0.2, max_value=5.0, value=1.0, step=0.01)
cut = st.selectbox("Cut", options=cut_options)
color = st.selectbox("Color", options=color_options)
clarity = st.selectbox("Clarity", options=clarity_options)
depth = st.slider("Depth (%)", min_value=40.0, max_value=80.0, value=60.0, step=0.1)
table = st.slider("Table (%)", min_value=40.0, max_value=80.0, value=55.0, step=0.1)
x = st.slider("X (mm)", min_value=0.0, max_value=10.0, value=5.0, step=0.01)
y = st.slider("Y (mm)", min_value=0.0, max_value=10.0, value=5.0, step=0.01)
z = st.slider("Z (mm)", min_value=0.0, max_value=10.0, value=5.0, step=0.01)

# Predict button
if st.button("Predict Price"):
    # Prepare input data
    input_data = pd.DataFrame({
        'carat': [carat],
        'cut': [le_cut.transform([cut])[0]],
        'color': [le_color.transform([color])[0]],
        'clarity': [le_clarity.transform([clarity])[0]],
        'depth': [depth],
        'table': [table],
        'x': [x],
        'y': [y],
        'z': [z]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    st.success(f"Predicted Diamond Price: ${prediction:,.2f}")

# Footer
st.markdown("---")
st.write("Built with Streamlit and a Random Forest model trained on diamond data.")