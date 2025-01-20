import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load the trained model
with open('machine.sav', 'rb') as file:
    model = pickle.load(file)

# Define the prediction function
def predict_heart_disease(data):
    prediction = model.predict(data)
    return prediction

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.title("Heart Disease Prediction")
st.image("heart.jpeg", width=200)

# Sidebar for user inputs
st.sidebar.header("Patient Data")
st.sidebar.image("heart.jpeg", width=200)
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=25, help="Enter the patient's age.")
sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help="Select the patient's sex.")
cp = st.sidebar.selectbox("Chest Pain Type", options=[0, 1, 2, 3], help="Type of chest pain experienced.")
trestbps = st.sidebar.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120, help="Resting blood pressure in mm Hg.")
chol = st.sidebar.number_input("Serum Cholesterol", min_value=100, max_value=600, value=200, help="Serum cholesterol in mg/dl.")
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], help="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).")
restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2], help="Resting electrocardiographic results.")
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150, help="Maximum heart rate achieved.")
exang = st.sidebar.selectbox("Exercise Induced Angina", options=[0, 1], help="Exercise induced angina (1 = yes; 0 = no).")
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, help="ST depression induced by exercise relative to rest.")
slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2], help="Slope of the peak exercise ST segment.")
ca = st.sidebar.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3, 4], help="Number of major vessels (0-3) colored by fluoroscopy.")
thal = st.sidebar.selectbox("Thalassemia", options=[0, 1, 2, 3], help="Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect).")

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Predict button
if st.sidebar.button("Predict"):
    prediction = predict_heart_disease(input_data)
    if prediction[0] == 1:
        st.success("The model predicts that the patient has heart disease.")
    else:
        st.success("The model predicts that the patient does not have heart disease.")

# Display input data
st.subheader("Patient Data")
st.write(input_data)

# Example chart
fig, ax = plt.subplots()
ax.hist(input_data['age'], bins=10 )
st.pyplot(fig)
