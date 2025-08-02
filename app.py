import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load model and encoders
# -----------------------------
with open('salary_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# -----------------------------
# Define input options
# -----------------------------
job_titles = label_encoders['job_title'].classes_.tolist()
experience_levels = label_encoders['experience_level'].classes_.tolist()
residences = label_encoders['employee_residence'].classes_.tolist()
company_locations = label_encoders['company_location'].classes_.tolist()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Salary Prediction App", layout="centered")
st.title("💼 Salary Prediction App")
st.write("Estimate employee salaries based on job title, experience level, and location.")

job = st.selectbox("Job Title", job_titles)
exp_level = st.selectbox("Experience Level", experience_levels)
residence = st.selectbox("Employee Residence", residences)
company_loc = st.selectbox("Company Location", company_locations)

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict Salary"):
    input_data = {
        'job_title': label_encoders['job_title'].transform([job])[0],
        'experience_level': label_encoders['experience_level'].transform([exp_level])[0],
        'employee_residence': label_encoders['employee_residence'].transform([residence])[0],
        'company_location': label_encoders['company_location'].transform([company_loc])[0]
    }

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Salary (USD): ${prediction:,.2f}")
