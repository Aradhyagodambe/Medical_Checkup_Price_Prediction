import streamlit as st
import joblib
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    # Ensure your model file is named 'model.pkl' and in the root directory
    return joblib.load('Checkup_Price.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'model.pkl' not found. Please upload it to your repository.")
    st.stop()

st.title("Insurance Cost Predictor 🏥")
st.write("This is a practice web app deployed on AWS via GitHub. Enter the details below to predict the insurance charge.")

# Define the user inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=120, value=30)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

with col2:
    # Mappings correspond to typical ordinal encoding for this dataset
    sex_input = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])
    smoker_input = st.selectbox("Smoker", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
    region_input = st.selectbox("Region", options=[("Southwest", 0), ("Southeast", 1), ("Northwest", 2), ("Northeast", 3)], format_func=lambda x: x[0])

if st.button("Predict Charge", type="primary"):
    # Model expects feature_names_in_: ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    features = np.array([[
        age, 
        sex_input[1], 
        bmi, 
        children, 
        smoker_input[1], 
        region_input[1]
    ]])
    
    prediction = model.predict(features)[0]
    
    st.success(f"**Predicted Insurance Cost:** ${prediction:,.2f}")
